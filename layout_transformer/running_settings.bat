python main_conditional.py --freeze_search_model False --search_loss True --device 'cuda:3' --exp 'search_loss_lr10e-6'

python main_conditional.py --freeze_search_model True --search_loss False --device 'cuda:1' --exp 'frozensearchmodel_lr10e-6'



======== Eval 


nohup python generate_conditional.py --pair 'different' --pt_model_path 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_False'> cond1_gen.out &
nohup python generate_conditional.py --pair 'different' --pt_model_path 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_True' > cond2_gen.out &
nohup python generate_conditional.py --pair 'same' --pt_model_path 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_True'      > cond3_gen.out &
nohup python generate_conditional.py --pair 'different' --pt_model_path 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMTrue_SL_False' --device 'cuda:1' > cond4_gen.out &
