from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# ข้อมูล SET50 จากไฟล์ SET50_100_H1_2026.pdf (อัปเดต H1/2026)
set50_companies = [
    {"symbol": "ADVANC", "name": "Advanced Info Service", "sector": "Information & Communication Technology"},
    {"symbol": "AOT", "name": "Airports of Thailand", "sector": "Transportation & Logistics"},
    {"symbol": "AWC", "name": "Asset World Corp", "sector": "Property Development"},
    {"symbol": "BANPU", "name": "Banpu", "sector": "Energy & Utilities"},
    {"symbol": "BBL", "name": "Bangkok Bank", "sector": "Banking"},
    {"symbol": "BDMS", "name": "Bangkok Dusit Medical Services", "sector": "Health Care Services"},
    {"symbol": "BEM", "name": "Bangkok Expressway and Metro", "sector": "Transportation & Logistics"},
    {"symbol": "BH", "name": "Bumrungrad Hospital", "sector": "Health Care Services"},
    {"symbol": "BJC", "name": "Berli Jucker", "sector": "Commerce"},
    {"symbol": "BTS", "name": "BTS Group Holdings", "sector": "Transportation & Logistics"},
    {"symbol": "CBG", "name": "Carabao Group", "sector": "Food & Beverage"},
    {"symbol": "CCET", "name": "Cal-Comp Electronics (Thailand)", "sector": "Electronic Components"},
    {"symbol": "CENTEL", "name": "Central Plaza Hotel", "sector": "Tourism & Leisure"},
    {"symbol": "COM7", "name": "Com7", "sector": "Commerce"},
    {"symbol": "CPALL", "name": "CP All", "sector": "Commerce"},
    {"symbol": "CPF", "name": "Charoen Pokphand Foods", "sector": "Food & Beverage"},
    {"symbol": "CPN", "name": "Central Pattana", "sector": "Property Development"},
    {"symbol": "CRC", "name": "Central Retail Corporation", "sector": "Commerce"},
    {"symbol": "DELTA", "name": "Delta Electronics (Thailand)", "sector": "Electronic Components"},
    {"symbol": "EGCO", "name": "Electricity Generating", "sector": "Energy & Utilities"},
    {"symbol": "GPSC", "name": "Global Power Synergy", "sector": "Energy & Utilities"},
    {"symbol": "GULF", "name": "Gulf Development", "sector": "Energy & Utilities"},
    {"symbol": "HMPRO", "name": "Home Product Center", "sector": "Commerce"},
    {"symbol": "IVL", "name": "Indorama Ventures", "sector": "Petrochemicals & Chemicals"},
    {"symbol": "KBANK", "name": "Kasikornbank", "sector": "Banking"},
    {"symbol": "KKP", "name": "Kiatnakin Phatra Bank", "sector": "Banking"},
    {"symbol": "KTB", "name": "Krung Thai Bank", "sector": "Banking"},
    {"symbol": "KTC", "name": "Krungthai Card", "sector": "Finance & Securities"},
    {"symbol": "LH", "name": "Land and Houses", "sector": "Property Development"},
    {"symbol": "MINT", "name": "Minor International", "sector": "Tourism & Leisure"},
    {"symbol": "MTC", "name": "Muangthai Capital", "sector": "Finance & Securities"},
    {"symbol": "OR", "name": "PTT Oil and Retail Business", "sector": "Energy & Utilities"},
    {"symbol": "OSP", "name": "Osotspa", "sector": "Food & Beverage"},
    {"symbol": "PTT", "name": "PTT Public Company", "sector": "Energy & Utilities"},
    {"symbol": "PTTEP", "name": "PTT Exploration and Production", "sector": "Energy & Utilities"},
    {"symbol": "PTTGC", "name": "PTT Global Chemical", "sector": "Petrochemicals & Chemicals"},
    {"symbol": "RATCH", "name": "Ratch Group", "sector": "Energy & Utilities"},
    {"symbol": "SAWAD", "name": "Srisawad Corporation", "sector": "Finance & Securities"},
    {"symbol": "SCB", "name": "SCB X", "sector": "Banking"},
    {"symbol": "SCC", "name": "Siam Cement Group", "sector": "Construction Materials"},
    {"symbol": "SCGP", "name": "SCG Packaging", "sector": "Packaging"},
    {"symbol": "TCAP", "name": "Thanachart Capital", "sector": "Banking"},
    {"symbol": "TIDLOR", "name": "Tidlor Holdings", "sector": "Finance & Securities"},
    {"symbol": "TISCO", "name": "TISCO Financial Group", "sector": "Banking"},
    {"symbol": "TLI", "name": "Thai Life Insurance", "sector": "Insurance"},
    {"symbol": "TOP", "name": "Thai Oil", "sector": "Energy & Utilities"},
    {"symbol": "TRUE", "name": "True Corporation", "sector": "Information & Communication Technology"},
    {"symbol": "TTB", "name": "TMBThanachart Bank", "sector": "Banking"},
    {"symbol": "TU", "name": "Thai Union Group", "sector": "Food & Beverage"},
    {"symbol": "WHA", "name": "WHA Corporation", "sector": "Property Development"}
]

all_shareholders_data = []

# ตั้งค่าให้เบราว์เซอร์ทำงาน (สามารถใส่ options.add_argument('--headless') เพื่อซ่อนหน้าจอได้)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

for company in set50_companies:
    symbol = company['symbol']
    company_name = company['name']
    sector = company['sector']
    
    url = f"https://www.set.or.th/th/market/product/stock/quote/{symbol}/major-shareholders"
    driver.get(url)
    
    print(f"กำลังเข้าถึงข้อมูลของ {symbol}...")
    
    try:
        wait = WebDriverWait(driver, 15)
        
        # 1. ล็อกเป้าหมาย: หา "ตาราง" ที่หัวตาราง (th) มีคำว่า "ผู้ถือหุ้น" อยู่ เพื่อให้ได้ตารางที่ถูกต้อง 100%
        target_table = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//table[.//thead//th[contains(., 'ผู้ถือหุ้น')]]")
        ))
        
        # 2. รออีกเล็กน้อยให้ Vue.js นำข้อมูลยัดลงแถวให้เสร็จ
        time.sleep(2)
        
        # 3. เจาะเข้าไปดึงแถว (tr) ในส่วนเนื้อหา (tbody) ของตารางนั้น
        rows = target_table.find_elements(By.XPATH, ".//tbody/tr")
        
        found_data = False
        
        # 4. วนลูปดึงข้อมูล 5 ลำดับแรก
        for row in rows[:5]:
            cols = row.find_elements(By.TAG_NAME, "td")
            
            if len(cols) >= 4:
                # *** จุดสำคัญ: เปลี่ยนจากการใช้ .text มาใช้ .get_attribute("textContent") ทะลวงการซ่อนของ CSS ***
                stakeholder = cols[1].get_attribute("textContent").strip()
                pct = cols[3].get_attribute("textContent").strip()
                
                if stakeholder:
                    all_shareholders_data.append({
                        'company': symbol,
                        'company_name': company_name,
                        'sector': sector,
                        'stakeholder': stakeholder,
                        'pct': pct
                    })
                    found_data = True
                    print(f"   -> {stakeholder} ({pct}%)")
        
        if not found_data:
            print(f" [!] ไม่พบข้อมูล (ตารางว่างเปล่า) ของ {symbol}")
            
    except Exception as e:
        print(f" [X] เกิดข้อผิดพลาดกับ {symbol}: หาตารางไม่พบหรือโหลดนานเกินไป")
    
    # หน่วงเวลาเพื่อไม่ให้ Request ถี่เกินไป
    time.sleep(3)

driver.quit()

# บันทึกข้อมูล
if len(all_shareholders_data) > 0:
    df = pd.DataFrame(all_shareholders_data)
    csv_filename = 'set50_stakeholders.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ เสร็จสิ้น! บันทึกข้อมูลทั้งหมด {len(all_shareholders_data)} รายการ ลงในไฟล์ {csv_filename} แล้ว")
else:
    print("\n❌ ไม่พบข้อมูลเลย ไฟล์ยังไม่ได้ถูกบันทึกทับครับ")