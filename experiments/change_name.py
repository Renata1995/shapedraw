import os

exp_dir = os.getcwd()
stm_dir = os.path.join(exp_dir, 'final_stimulus')
for index, file in enumerate(os.listdir(stm_dir)[1:]):
    num = int(index/4) + 1
    new_filename = '00' + str(num) + file[3:]
    os.rename(os.path.join(stm_dir,file), os.path.join(stm_dir,new_filename))