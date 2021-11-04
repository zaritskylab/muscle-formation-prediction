import gdown
import os
import glob
if __name__ == "__main__":

    dir = "/home/shakarch/muscle-formation-diff/data/images/all_images"

    for i in range(10):
        for j in range(10):

            for tifpath in glob.iglob(os.path.join(dir, f'*{i}{j}.tif')):
                if os.path.exists(tifpath):
                    print(tifpath)
                    os.remove(tifpath)







    # urls = {
    #     "s3_Nuclei": 'https://drive.google.com/uc?id=1c9tq21ClbXtZ-Y0WrGwV5ZPaaw5UCf_e',  # control
    #         "s5_Nuclei": "https://drive.google.com/uc?id=1aWyiUfyumXxmXNrKfv0OYtq914XCU6fn",  # control
    #         "s8_Nuclei": "https://drive.google.com/uc?id=11L6fzT8-K_a4g-6nZCvBHGYFzPYqCPrI",  # ERKi
    #         "s10_Nuclei": "https://drive.google.com/uc?id=1S0pPD76w86wGW-vir0H_4Wxlk564lCl7",  # ERKi
    #         }
    # # url = 'https://drive.google.com/uc?id=1c9tq21ClbXtZ-Y0WrGwV5ZPaaw5UCf_e'
    #
    # for url in urls:
    #     # output = 'C:/Users/Amit/PycharmProjects/muscle-formation-regeneration/s3_Nuclei.tif'
    #     output = f'/home/shakarch/muscle-formation-diff/data/videos/210726/{url}.tif'
    #     gdown.download(urls[url], output, quiet=False)
