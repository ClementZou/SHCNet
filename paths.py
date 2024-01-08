from option import opt
import os

train_path = os.path.join(opt.train_path_root, opt.folder_name) + '/'
valid_path = os.path.join(opt.valid_path_root, opt.folder_name) + '/'
test_path = os.path.join(opt.test_path_root, opt.folder_name) + '/'
test_path_real = os.path.join(opt.real_path_root, opt.folder_name_test) + '/'
loss_string = '&'.join(f"{word}" for word in opt.loss)
sim_flag = "sim" if opt.sim else "real"
model_name = f"channel_{opt.channel}_" \
             f"lr_{opt.learning_rate}_" \
             f"rho_{opt.rho}_" \
             f"stage_{opt.stage}_" \
             f"set_{opt.folder_name}_" \
             f"{sim_flag}_" \
             f"loss_{loss_string}_" \
             f"scheduler_{opt.scheduler}"
if opt.flag != '':
    model_name = f"{model_name}_flag_{opt.flag}"
if opt.bias_index != -1:
    model_name = f"{model_name}_index_{opt.bias_index}"
finetune_path = os.path.join(opt.model_path_root, opt.method, model_name + 'fine_tune') + '/'
model_path = os.path.join(opt.model_path_root, opt.method, model_name) + '/'
discriminator_path = os.path.join(opt.model_path_root, opt.method, model_name) + '_discriminator/'
result_path = os.path.join(opt.result_path_root, opt.folder_name_test, opt.method, model_name, f"epoch_{opt.test_epoch}") + '/'
if opt.first_epoch != 0:
    pretrained_model_path = os.path.join(model_path, f"model_{opt.first_epoch:03d}.pth")
else:
    pretrained_model_path = None

debug_num = 0
