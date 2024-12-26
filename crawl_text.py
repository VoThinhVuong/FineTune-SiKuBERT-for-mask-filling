from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import time
import csv
import re




def is_cjk(char):
    if char == '冫' or char == "“" or char == "”":
        return False
    return any([
        '\u4E00' <= char <= '\u9FFF',  # CJK Unified Ideographs
        '\u3400' <= char <= '\u4DBF',  # CJK Unified Ideographs Extension A
        '\u20000' <= char <= '\u2A6DF', # CJK Unified Ideographs Extension B
        '\u2A700' <= char <= '\u2EBEF', # CJK Unified Ideographs Extension C-F
        '\uF900' <= char <= '\uFAFF',  # CJK Compatibility Ideographs
    ])

def contains_cjk(text):
    return any(is_cjk(char) for char in text)



def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")            
    chrome_options.add_argument("--disable-gpu")         
    chrome_options.add_argument("--no-sandbox")          
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================

def crawl_pages_Kieu1992(driver, url, total_pages=5):
    text_data = []  

    try:
        driver.get(url)
        time.sleep(3)  

        for current_page in range(total_pages):
            print(f"Processing page {current_page + 1}...")

            page_source = driver.page_source

            soup = BeautifulSoup(page_source, 'html.parser')

            br_elements = soup.find_all('br')

            for br in br_elements:
                text_before_br = br.previous_sibling
                if text_before_br and isinstance(text_before_br, str):
                    cleaned_text = text_before_br.strip()  

                    if contains_cjk(cleaned_text):
                        text_data.append(cleaned_text)  

            if current_page < total_pages - 1:  
                page_number = current_page + 1 
                driver.execute_script(f"javascript:GotoPage({page_number})")
                time.sleep(3)  

    except Exception as e:
        print(f"An error occurred: {e}")

    return text_data


def crawling_Kieu1902(max_page = 163):
    url = 'https://nomfoundation.org/nom-project/tale-of-kieu/tale-of-kieu-version-1902?uiLang=en'
    output_file = 'Kieu1902.txt'
    driver = create_driver()

    try:
        text_data = crawl_pages_Kieu1992(driver, url, total_pages=max_page)

        with open(output_file, mode='w', encoding='utf-8') as file:
            for text in text_data:
                file.write(text + '\n')  

        print(f"Dữ liệu đã được lưu vào {output_file}")
    finally:
        driver.quit()
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================



def crawl_pages_LucVanTien(driver, url, total_pages=5):
    text_data = []  # List to store texts containing CJK characters

    try:
        driver.get(url)
        time.sleep(3)  # Wait for the page to load completely

        for current_page in range(total_pages):
            print(f"Processing page {current_page + 1}...")

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Lấy tất cả các thẻ <tr> chứa thẻ <td class="hnText">
            hn_text_elements = soup.find_all('td', class_='hnText')

            for element in hn_text_elements:
                # Tìm tất cả các thẻ <br> trong mỗi phần tử
                br_elements = element.find_all('br')

                for br in br_elements:
                    text_before_br = br.previous_sibling
                    if text_before_br and isinstance(text_before_br, str):
                        cleaned_text = text_before_br.strip()  # Lấy văn bản trước thẻ <br>

                        if contains_cjk(cleaned_text):
                            text_data.append(cleaned_text)  # Nếu chứa CJK, thêm vào danh sách

            if current_page < total_pages - 1:  # Chuyển trang nếu còn trang kế tiếp
                page_number = current_page + 1  # Tính số trang tiếp theo
                driver.execute_script(f"javascript:GotoPage({page_number})")
                time.sleep(3)  # Đợi trang tiếp theo tải xong

    except Exception as e:
        print(f"An error occurred: {e}")

    return text_data



def crawling_LucVanTien(max_page = 104):
    url = 'https://nomfoundation.org/nom-project/Luc-Van-Tien/Luc-Van-Tien-Text?uiLang=en'
    output_file = 'LucVanTien.txt'
    driver = create_driver()

    try:
        text_data = crawl_pages_LucVanTien(driver, url, total_pages=max_page)

        # Lưu dữ liệu vào file
        with open(output_file, mode='w', encoding='utf-8') as file:
            for text in text_data:
                file.write(text + '\n')  # Lưu mỗi đoạn văn vào file

        print(f"Dữ liệu đã được lưu vào {output_file}")
    finally:
        driver.quit()




#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================

def crawl_pages_ChinhPhuNgamKhuc(driver, url, total_pages=5):
    text_data = []  # List to store texts containing CJK characters

    try:
        driver.get(url)
        time.sleep(3)  # Wait for the page to load initially

        for current_page in range(total_pages):
            print(f"Processing page {current_page + 1}...")

            # Wait for the page content to load by checking a specific element
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".hnText"))
            )

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Tìm bảng có chứa tiêu đề "NomText"
            nom_text_table = soup.find('b', text='NomText')  # Tìm thẻ <b> có chứa "NomText"

            if nom_text_table:
                print(nom_text_table)
                # Tìm bảng chứa thẻ <b> NomText
                table = nom_text_table.find_parent('table')

                # Lấy tất cả các thẻ <tr> trong table chứa "NomText"
                tr_elements = table.find_all('tr')

                for tr in tr_elements:
                    # Lấy tất cả các thẻ <td class="hnText">
                    hn_text_elements = tr.find_all('td', class_='hnText')

                    for element in hn_text_elements:
                        # Tìm tất cả các thẻ <br> trong mỗi phần tử <td class="hnText">
                        br_elements = element.find_all('br')
                        for br in br_elements:
                            text_before_br = br.previous_sibling
                            if text_before_br and isinstance(text_before_br, str):
                                cleaned_text = text_before_br.strip()  # Lấy văn bản trước thẻ <br>

                                if contains_cjk(cleaned_text):
                                    text_data.append(cleaned_text)  # Nếu chứa CJK, thêm vào danh sách

            if current_page < total_pages - 1:  # Chuyển trang nếu còn trang kế tiếp
                page_number = current_page + 1  # Tính số trang tiếp theo
                driver.execute_script(f"javascript:GotoPage({page_number})")
                time.sleep(3)  # Đợi trang tiếp theo tải xong


    except Exception as e:
        print(f"An error occurred: {e}")

    return text_data



def crawling_ChinhPhuNgamKhuc(max_page= 64):
    url = 'https://nomfoundation.org/nom-project/Chinh-Phu-Ngam-Khuc/Chinh-Phu-Ngam-Khuc-Text/Chinh-Phu-Ngam-Khuc-Text?uiLang=en'
    output_file = 'ChinhPhuNgamKhuc.txt'
    driver = create_driver()

    try:
        text_data = crawl_pages_ChinhPhuNgamKhuc(driver, url, total_pages=max_page)

        with open(output_file, mode='w', encoding='utf-8') as file:
            for text in text_data:
                file.write(text + '\n')  

        print(f"Dữ liệu đã được lưu vào {output_file}")
    finally:
        driver.quit()



#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================


def crawl_pages_QuocAmThiTap(url_template, total_pages=254):
    all_texts = []
    
    for page_num in range(1, total_pages + 1):
        url = url_template.format(page_num)
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tìm tất cả các thẻ <td> có class="hnText", align="justify", width="30%" và valign="top"
            td_elements = soup.find_all('td', class_='hnText')
            # Lấy text trong từng thẻ <td>
            for td in td_elements:
                text = td.get_text(strip=True)
                if contains_cjk(text):
                    all_texts.append(text)
        else:
            print(f"Failed to retrieve page {page_num}")
    
    return all_texts

def crawling_QuocAmThiTap(number_pages=254):
    url_template = "https://nomfoundation.org/nom-tools/QATT/QATT/?qatt_group_idx=0&poem_id={}&uiLang=en"
    output_file = 'QuocAmThiTap.txt'
    
    # Crawl dữ liệu từ tất cả các trang
    text_data = crawl_pages_QuocAmThiTap(url_template, total_pages=number_pages)
    
    # Lưu kết quả vào file
    with open(output_file, mode='w', encoding='utf-8') as file:
        for text in text_data:
            file.write(text + '\n')

    print(f"Dữ liệu đã được lưu vào {output_file}")

def extract_han_nom(text):
    # Sử dụng regex để tìm các ký tự Hán-Nôm, bao gồm cả các ký tự trong các bảng chữ Hán cổ và hiện đại
    han_nom_pattern = r'[\u4e00-\u9fff\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF]+'

    # Tìm tất cả các chuỗi Hán-Nôm trong văn bản
    matches = re.findall(han_nom_pattern, text)

    # Nếu có kết quả, lấy phần tử đầu tiên
    if matches:
        return matches[0]
    return ''


def main():

    # # Maximum: 163
    # crawling_Kieu1902(163)



    # # Maximum: 104
    # crawling_LucVanTien(104)  


    # # Maximum: 64
    # crawling_ChinhPhuNgamKhuc(64)


    # # Maximum: 254
    # crawling_QuocAmThiTap(254)
    text1 = "Thử thủy kỷ thời thể   此水幾时体"
    text2 = " Ðược    特貝暈𦝄𠳨   với vầng trăng hỏi "

    han_nom1 = extract_han_nom(text1)
    han_nom2 = extract_han_nom(text2)

    print(f"Chuỗi chữ Hán-Nôm trong văn bản 1: {han_nom1}")
    print(f"Chuỗi chữ Hán-Nôm trong văn bản 2: {han_nom2}")



if __name__ == "__main__":
    main()