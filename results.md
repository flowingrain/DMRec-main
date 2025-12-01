# amazon
python train_encoder.py --base_model mult_vae --strategy godm --dataset amazon --cuda 0

Best Epoch 144. Final test result: {'recall': array([0.06668494, 0.10466223, 0.15347752]), 'ndcg': array([0.06744571, 0.08060764, 0.09665725])}.

python train_encoder.py --model mult_vae_mddm --dataset amazon --cuda 0
Best Epoch 183. Final test result: {'recall': array([0.06974205, 0.10507024, 0.15604424]), 'ndcg': array([0.06813704, 0.08042316, 0.09756433])}.

python train_encoder.py --base_model mult_vae --strategy fmdm --dataset amazon --cuda 0
Best Epoch 120. Final test result: {'recall': array([0.10166969, 0.14958132]), 'ndcg': array([0.07693701, 0.0928937 ])}.

python train_encoder.py --base_model mult_vae --strategy rfdm --dataset amazon --cuda 0
Best Epoch 102. Final test result: {'recall': array([0.10476952, 0.15300632]), 'ndcg': array([0.07929117, 0.09523369])}.

