import subprocess
failure = 1
counter = 0
while failure:
    print("run number: " + str(counter))
    result = subprocess.run("python PDL1_main.py test --dataset D:\\Nati\\Itamar_n_Dekel\\data\\full_slide --weights D:\\Nati\\Itamar_n_Dekel\\pathologylab\\good_weights\\aug_img1-256_medium_anchor_all-th-06_50roi_roi-ratio-04_class-loss3_smaller-lr_more-epochs_0020.h5 --real --result_dir D:\\Nati\\Itamar_n_Dekel\\pathologylab\\output\\out_20211020T225007_real_slide")
    #"test", "--dataset D:\\Nati\\Itamar_n_Dekel\\data\\full_slide", "--weights D:\\Nati\\Itamar_n_Dekel\\pathologylab\\good_weights\\aug_img1-256_medium_anchor_all-th-06_50roi_roi-ratio-04_class-loss3_smaller-lr_more-epochs_0020.h5", "--real", "--result_dir D:\\Nati\\Itamar_n_Dekel\\pathologylab\\output\\out_20211020T225007_real_slide"])
    failure = result.returncode

