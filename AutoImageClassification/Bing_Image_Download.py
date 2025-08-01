# Web scraping and downloading imgaes from Bing
import os
from PIL import Image
import glob
from icrawler.builtin import BingImageCrawler

class Bing_Image_Download:
    def __init__(self, Download_num, Output_folder, Categories, Keywords = None):
        self.Download_num = Download_num
        self.Output_folder = Output_folder
        self.Categories = Categories
        # By default it uses the category names as search keywords
        if not Keywords:
            Keywords = Categories
        self.Keywords = Keywords

    # This function download images from Bing with _keyword search and saved them into directory_name
    # The function return the result after sending _request_num requests or reaching _max_num of images downloaded 
    def Image_Download(self, _request_num, _max_num, _keyword, directory_name):
        directory_path = f'{self.Output_folder}/{directory_name}'
    
        for i in range(_request_num):
            crawler = BingImageCrawler(storage={'root_dir': directory_path})
            crawler.crawl(keyword=_keyword, max_num=_max_num)
            file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
            if file_count >= _max_num:
                break
    
        # Removing corrupted images
        image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Check if image is corrupted
            except Exception as e:
                print(f"Removing corrupted image: {image_path} ({e})")
                os.remove(image_path)

    # This function download images for all the categories. It sends request to bing server 10 times
    def Downloading(self):
        for index in range(0, len(self.Categories)):
            self.Image_Download(10,self.Download_num,self.Keywords[index],self.Categories[index])


