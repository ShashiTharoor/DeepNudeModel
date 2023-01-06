import sys
import cv2
from run import process
import argparse
import os


def main():
    for file in os.listdir("input_images"):
        i_image=f"input_images/{file}"
        o_image=f"output_images/{file}"
        _process(i_image, o_image, False)

def main2():
    parser = argparse.ArgumentParser(description='DeepNude App CLI Version with no Watermark.')
    parser.add_argument('-i', "--input", help='Input image to process.', action="store", dest="input", required=True)
    parser.add_argument('-o', "--output",help='Output path to save result.', action="store", dest="output", required=False, default="output.jpg")
    parser.add_argument('-g', "--use-gpu", help='Enable using CUDA gpu to speed up the process.', action="store_true",dest="use_gpu", default=False)
    
    if not os.path.isdir("checkpoints"):
        pcv_imgrint("[-] Checkpoints folder not found, download it from Github repository, and extract files to 'checkpoints' folder.")
        sys.exit(1)
    arguments = parser.parse_args()
    
    print("[*] Processing: %s" % arguments.input)
    
    if (arguments.use_gpu):
        print("[*] Using CUDA gpu to speed up the process.")
    
    _process(arguments.input, arguments.output, arguments.use_gpu)


def _process(i_image, o_image, use_gpu):
    try:
        dress = cv2.imread(i_image)
        h = dress.shape[0]
        w = dress.shape[1]
        dress = cv2.resize(dress, (512,512), interpolation=cv2.INTER_CUBIC)
        watermark = process(dress, use_gpu)
        watermark =  cv2.resize(watermark, (w,h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(o_image, watermark)
        print("[*] Image saved as: %s" % o_image)
    except Exception as ex:
        ex = str(ex)
        if "NoneType" in ex:
            print("[-] File %s not found" % i_image)
        elif "runtime error" in ex:
            print("[-] Error: CUDA Runtime not found, Disable the '--use-gpu' option!")
        else:
            print("[-] Error occured when trying to process the image: %s" % ex)
            with open("logs.txt", "a") as f:
                f.write("[-] Error: %s\n" % ex)
        # sys.exit(1)

if __name__ == '__main__':
    main()
