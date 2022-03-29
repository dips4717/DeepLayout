import torch
from torch import nn

from gpt2 import Block, CondBlock, Conv1D

class GPT2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.seq_vocab_sizes) < 1:
            raise RuntimeError('Must provide one vocabulary size for each sequence.')
        if len(config.output_seq_dims) < 1:
            raise RuntimeError('Must provide dimensions for at least one output sequence.')

        self.sequence_embeddings = nn.ModuleList()
        for seq_vocab_size in config.seq_vocab_sizes:
            if seq_vocab_size is None:
                self.sequence_embeddings.append(None)
            else:
                self.sequence_embeddings.append(nn.Embedding(num_embeddings=seq_vocab_size, embedding_dim=config.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)
        if config.conditional:
            self.h = nn.ModuleList([CondBlock(n_ctx=config.n_positions, config=config, scale=True) for _ in range(config.n_layer)])
            if config.cond_vocab_size is not None:
                self.cond_embedding = nn.Embedding(num_embeddings=config.cond_vocab_size, embedding_dim=config.n_embd)
            else:
                self.cond_embedding = None
        else:
            self.h = nn.ModuleList([Block(n_ctx=config.n_positions, config=config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, sum(config.output_seq_dims))

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.sequence_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.sequence_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        sequences,
        cond=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past is None:
            past = [None] * len(self.h)

        if labels is not None and len(labels) != len(self.config.output_seq_dims):
            raise RuntimeError(f'Expected {len(self.config.output_seq_dims)} label sequences but got {len(labels)}.')
        
        if len(sequences) != len(self.sequence_embeddings):
            raise RuntimeError(f'Expected {len(self.sequence_embeddings)} sequences but got {len(sequences)}.')
        if sequences[0].ndim < 2:
            raise RuntimeError('Sequences must be of shape batch size x sequence length [x embedding dimension (only when passing embedded sequences)].')
        if sequences[0].shape[0] == 0:
            raise RuntimeError('Batch size must be > 0.')

        batch_size = sequences[0].shape[0]
        seq_len = sequences[0].shape[1]

        hidden_states = torch.zeros(batch_size, seq_len, self.config.n_embd, device=sequences[0].device, dtype=sequences[0].dtype)
        for si, sequence in enumerate(sequences):
            if self.sequence_embeddings[si] is None:
                if sequence.shape != (batch_size, seq_len, self.config.n_embd):
                    raise RuntimeError('Embedded sequences must have the shape (batch size x sequence length x embedding dimension).')
                hidden_states = hidden_states + sequence
            else:
                if sequence.shape != (batch_size, seq_len):
                    raise RuntimeError('Sequences must have the shape (batch size x sequence length).')
                hidden_states = hidden_states + self.sequence_embeddings[si](sequence)

        if self.config.conditional and self.cond_embedding is not None:
            cond = self.cond_embedding(cond)

        # attention mask
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise RuntimeError('The sequence mask must have shape  (batch size x max. sequence length)')
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            if attention_mask.dtype != torch.int32:
                raise RuntimeError('Currently only int32 attention masks are supprted.')
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = head_mask = [None] * self.config.n_layer #self.get_head_mask(head_mask, self.config.n_layer)

        # dropout
        hidden_states = self.drop(hidden_states)

        output_shape = (batch_size, seq_len, hidden_states.size(-1))

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
            if self.config.conditional:
                outputs = block(
                    hidden_states,
                    cond=cond,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)

        hidden_states = hidden_states.view(batch_size, seq_len, sum(self.config.output_seq_dims))

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # split output into separate sequences
        hidden_states = torch.split(hidden_states, self.config.output_seq_dims, dim=-1)
        
        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (batch_size, -1) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        if labels is not None:
            loss = torch.zeros(batch_size, len(hidden_states), seq_len-1, dtype=hidden_states[0].dtype, device=hidden_states[0].device)
            for si in range(len(hidden_states)):
                # shift since ground truth output is the input sequence shifted one step to the left
                output_seq_logits = hidden_states[si][:, :-1, :].contiguous() # omit last token (always a padded token past the end of the sequence)
                label_seq = labels[si][:, 1:].contiguous() # omit start token (the start token is never an output of the transformer)
                loss[:, si, :] = nn.functional.cross_entropy(
                    input=output_seq_logits.view(-1, output_seq_logits.size(-1)),
                    target=label_seq.view(-1),
                    reduction='none').view(batch_size, seq_len-1)
            outputs = (loss,) + outputs

        return outputs  # (loss[optional], last hidden state, presents, all hidden_states, all_attentions)
