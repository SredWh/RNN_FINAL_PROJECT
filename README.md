下載ILSVRC2012 資料集產生對抗式樣本
models資料夾下載cnn模型的權重:
https://github.com/ylhz/tf_to_pytorch_model

修改attack.py 第12行
Ex: import methods_TGRv1 as  methods (用TGRv1產生對抗式樣本) 

python attack.py --attack TGR --batch_size 1 --model_name vit_base_patch16_224

評估對抗式樣本對vit模型的攻擊率
bash run_evaluate.sh model_vit_base_patch16_224-method_TGR

評估對抗式樣本對cnn模型的攻擊率
python evaluate_cnn.py

Code refer to: Towards Transferable Adversarial Attacks on Vision Transformers , tf_to_torch_model and TGR
