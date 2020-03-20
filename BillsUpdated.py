import pytesseract
import cv2
import math
import numpy as np
import os
import re

try:
    from PIL import Image
except ImportError:
    import Image

#from invoice2data import extract_data
import nltk
filename= 'dataset.txt'
s_w= [line.rstrip('\n') for line in open(filename)]
from nltk.tokenize import sent_tokenize, word_tokenize
stopwords = nltk.corpus.stopwords.words('english')
s_w= [line.rstrip('\n') for line in open(filename)]

v= ['Tel','Fac','FAX','TEL', 'Tax', 'Invoice', 'Email', '@', 'www.','GSTIN' , 'GST','.co.', ' CIN: ','600042','cri: T6.', '__.', ". '", 'www.', 'zaral com/in.', '.', 'GStIN: -~.']
for i in range(0, len(s_w)):
    stopwords.append(s_w[i])

shop_idt= (['Zara', 'Xiaomi','Wow Momos','VB SIGNATURE','Toys R Us', 'Starmark', 'Starbucks','Smoke- The Sizzler House','Shoppers Stop','RMKV','Reliance Trends','Put Kadalai'
                  ,'PMC Exit Towards Courtyard','Pizza Hut','PATISSEZ','Pantaloons','Mobile Zone','Miniso','Metro','Max','Marks and Spencer','Market99','Lyfe by Soul Garden Bistro',
                  'Luxe','Lifestyle','Latt Liv','Krispy Kreme','KFC.','Jockey','Home Centre','Health and Glow','H&M','Globus','- Galito\'s','Fun City','Dunkin Donuts','Dominos','Croma',' Sapphire Foods India Pt.. Ltd'
                  'Cold Stone Creamery','Burger King','Big Bazaar','Belgian Waffle','Bata','Archies','Annavillas','1st Step','Jubilant FoodWorks Limited','CROMA','Infiniti Retail Linited Trading', 'WOW MOMO FOODS.','ZARA', 'XIAOMI','WOW MOMOS','VB SIGNATURE','TOYS R US', 'STARMARK', 'STARBUCKS','SMOKE- THE SIZZLER HOUSE','SHOPPERS STOP','RMKV','RELIANCE TRENDS','PUT KADALAI'
                  ,'PMC EXIT TOWARDS COURTYARD','PIZZA HUT','PATISSEZ','PANTALOONS','MOBILE ZONE','MINISO','METRO','MAX','MARKS AND SPENCER','MARKET99','LYFE BY SOUL GARDEN BISTRO',
                  'LUXE','LIFESTYLE','LATT LIV','KRISPY KREME','KFC.','JOCKEY','HOME CENTRE','HEALTH AND GLOW','H&M','GLOBUS','- GALITO\'S','FUN CITY','DUNKIN DONUTS','DOMINOS','CROMA',' SAPPHIRE FOODS INDIA PT.. LTD'
                  'COLD STONE CREAMERY','BURGER KING','BIG BAZAAR','BELGIAN WAFFLE','BATA','ARCHIES','ANNAVILLAS','1ST STEP','JUBILANT FOODWORKS LIMITED','CROMA','INFINITI RETAIL LINITED TRADING', 'WOW MOMO FOODS.','Zara', 'Xiaomi','Wow Momos','Vb Signature','Toys R Us', 'Starmark', 'Starbucks','Smoke- The Sizzler House','Shoppers Stop','Rmkv','Reliance Trends','Put Kadalai'
                  ,'Pmc Exit Towards Courtyard','Pizza Hut','Patissez','Pantaloons','Mobile Zone','Miniso','Metro','Max','Marks And Spencer','Market99','Lyfe By Soul Garden Bistro',
                  'Luxe','Lifestyle','Latt Liv','Krispy Kreme','Kfc.','Jockey','Home Centre','Health And Glow','H&M','Globus','- Galito\'S','Fun City','Dunkin Donuts','Dominos','Croma',' Sapphire Foods India Pt.. Ltd'
                  'Cold Stone Creamery','Burger King','Big Bazaar','Belgian Waffle','Bata','Archies','Annavillas','1st Step','Jubilant Foodworks Limited','Croma','Infiniti Retail Linited Trading', 'Wow Momo Foods.'])

curr_path= 'C:'
folder = os.path.join(curr_path, '\dataset')
#path= r'C:\dataset2\zxcv.jpg'
for filename in os.listdir(folder):
    img = cv2.imread((os.path.join(folder, filename)))
    #img=cv2.imread(path)
    img_h = cv2.imread((os.path.join(folder, filename)))
    #img_h=cv2.imread(path)
    img_c = cv2.imread((os.path.join(folder, filename)))
    #img_c=cv2.imread(path)
    cv2.namedWindow('thresholded',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thresholded', 700, 700)
    cv2.namedWindow('header',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('header', 700, 700)
    cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('header', 500, 500)

    n = 1 # N. . .
    p,q= img.shape[:2]

    for i in range (int(0.27*p),p):
        for j in range(0,q):
               img_h[i,j]=0

    for i in range (0,int(0.27*p)):
        for j in range(0,q):
               img_c[i,j]=0



    org_imgh = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)
    org_imgc= cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    org_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #org_imgh= cv2.resize(org_imgh, (896, 500))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(org_imgh)
    cl2 = clahe.apply(org_imgc)
    cl3 = clahe.apply(org_img)




    cv2.imshow("thresholded",img)
    cv2.imshow("header", org_imgh)

    coords = cv2.findNonZero(cl3)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = cl3[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    cv2.imshow("Cropped", rect)  # Show it

    cv2.imwrite("header.png", cl1)



    v = cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("header.png",cl1)
    header_text = pytesseract.image_to_string(Image.open("header.png"))
    print(header_text)


    #content_text = pytesseract.image_to_string(Image.open("content.png"))
    #print(content_text)

    ex = header_text
    ex = ex.replace('\n', '. ')
    words = word_tokenize(ex)
    print(words)
    sentence = sent_tokenize(ex)

    #sentence = [s.replace('\n', '.') for s in sentence]

    print(sentence)
    #print(word_tokenize(ex))
    #print(sentence[0])

    p= ['LTD', 'Limited', 'Linited', 'LIMITED','Lim','Pvt','Put','Ltd', 'Retail']
    for i in range(0, len(p)):
        shop_idt.append(p[i])

    #filename='C:\City_dataset.txt'

    chr=[',','-']
    #lineList = [line.rstrip('\n') for line in open(filename)]
    lineList = ['Abohar ', 'Achalpur ', 'Adilabad ', 'Adityapur ', 'Adoni ', 'Agartala ', 'Agra ', 'Ahmadabad ', 'Ahmadnagar ', 'Aizawl ', 'Ajmer ', 'Akbarpur ', 'Akola ', 'Alandur ', 'Alappuzha ', 'Aligarh ', 'Allahabad ', 'Alwar ', 'Ambala ', 'Ambala Sadar ', 'Ambarnath ', 'Ambattur ', 'Ambikapur ', 'Ambur ', 'Amravati ', 'Amreli ', 'Amritsar ', 'Amroha ', 'Anand ', 'Anantapur ', 'Anantnag ', 'Arrah ', 'Asansol ', 'Ashoknagar Kalyangarh ', 'Aurangabad ', 'Aurangabad ', 'Avadi ', 'Azamgarh ', 'Badlapur ', 'Bagaha ', 'Bagalkot ', 'Bahadurgarh ', 'Baharampur ', 'Bahraich ', 'Baidyabati ', 'Baleshwar Town ', 'Ballia ', 'Bally ', 'Bally City', 'Balurghat ', 'Banda ', 'Bankura ', 'Bansberia ', 'Banswara ', 'Baran ', 'Baranagar ', 'Barasat ', 'Baraut ', 'Barddhaman ', 'Bareilly ', 'Baripada Town ', 'Barnala ', 'Barrackpur ', 'Barshi ', 'Basirhat ', 'Basti ', 'Batala ', 'Bathinda ', 'Beawar ', 'Begusarai ', 'Belgaum ', 'Bellary ', 'Bengaluru', 'Bettiah ', 'Betul ', 'Bhadrak ', 'Bhadravati ', 'Bhadreswar ', 'Bhagalpur ', 'Bhalswa Jahangir Pur ', 'Bharatpur ', 'Bharuch ', 'Bhatpara ', 'Bhavnagar ', 'Bhilai Nagar ', 'Bhilwara ', 'Bhimavaram ', 'Bhind ', 'Bhiwadi ', 'Bhiwandi ', 'Bhiwani ', 'Bhopal ', 'Bhubaneswar Town ', 'Bhuj ', 'Bhusawal ', 'Bid ', 'Bidar ', 'Bidhan Nagar ', 'Biharsharif ', 'Bijapur ', 'Bikaner ', 'Bilaspur ', 'Bokaro Steel City ', 'Bongaon ', 'Botad ', 'Brahmapur Town ', 'Budaun ', 'Bulandshahr ', 'Bundi ', 'Burari ', 'Burhanpur ', 'Buxar ', 'Champdani ', 'Chandannagar ', 'Chandausi ', 'Chandigarh ', 'Chandrapur ', 'Chapra ', 'Chas ', 'Chennai ', 'Chhattarpur ', 'Chhindwara ', 'Chikmagalur ', 'Chilakaluripet ', 'Chitradurga ','Nadu' 'CHen-Phoenix', 'Chittaurgarh ', 'Chittoor ', 'Churu ', 'Coimbatore ', 'Cuddalore ', 'Cuttack ', 'Dabgram ', 'Dallo Pura ', 'Damoh ', 'Darbhanga ', 'Darjiling ', 'Datia ', 'Davanagere ', 'Deesa ', 'Dehradun ', 'Dehri ', 'Delhi ', 'Delhi Cantonment ', 'Deoghar ', 'Deoli ', 'Deoria ', 'Dewas ', 'Dhanbad ', 'Dharmavaram ', 'Dhaulpur ', 'Dhule ', 'Dibrugarh ', 'Dimapur ', 'Dinapur Nizamat ', 'Dindigul ', 'Dum Dum ', 'Durg ', 'Durgapur ', 'Eluru ', 'English Bazar ', 'Erode ', 'Etah ', 'Etawah ', 'Faizabad ', 'Faridabad ', 'Farrukhabad-cum-Fatehgarh ', 'Fatehpur ', 'Firozabad ', 'Firozpur ', 'Gadag-Betigeri ', 'Gandhidham ', 'Gandhinagar ', 'Ganganagar ', 'Gangapur City ', 'Gangawati ', 'Gaya ', 'Ghazipur ', 'Giridih ', 'Godhra ', 'Gokal Pur ', 'Gonda ', 'Gondal ', 'Gondiya ', 'Gorakhpur ', 'Greater Hyderabad ', 'Greater Mumbai ', 'Greater Noida ', 'Gudivada ', 'Gulbarga ', 'Guna ', 'Guntakal ', 'Guntur ', 'Gurgaon ', 'Guwahati ', 'Gwalior ', 'Habra ', 'Hajipur ', 'Haldia ', 'Haldwani-cum-Kathgodam ', 'Halisahar ', 'Hanumangarh ', 'Haora ', 'Hapur ', 'Hardoi ', 'Hardwar ', 'Hassan ', 'Hastsal ', 'Hathras ', 'Hazaribag ', 'Hindaun ', 'Hindupur ', 'Hinganghat ', 'Hisar ', 'Hoshangabad ', 'Hoshiarpur ', 'Hospet ', 'Hosur ', 'Hubli-Dharwad ', 'Hugli-Chinsurah ', 'Ichalkaranji ', 'Imphal ', 'Indore ', 'Jabalpur ', 'Jagadhri ', 'Jagdalpur ', 'Jaipur ', 'Jalandhar ', 'Jalgaon ', 'Jalna ', 'Jalpaiguri ', 'Jamalpur ', 'Jammu ', 'Jamnagar ', 'Jamshedpur ', 'Jamuria ', 'Jaunpur ', 'Jehanabad ', 'Jetpur Navagadh ', 'Jhansi ', 'Jhunjhunun ', 'Jind ', 'Jodhpur ', 'Junagadh ', 'Kadapa ', 'Kaithal ', 'Kakinada ', 'Kalol ', 'Kalyani ', 'Kamarhati ', 'Kancheepuram ', 'Kanchrapara ', 'Kanpur ', 'Kanpur City', 'Karaikkudi ', 'Karawal Nagar ', 'Karimnagar ', 'Karnal ', 'Kasganj ', 'Kashipur ', 'Katihar ', 'Khammam ', 'Khandwa ', 'Khanna ', 'Kharagpur ', 'Khardaha ', 'Khargone ', 'Khora ', 'Khurja ', 'Kirari Suleman Nagar ', 'Kishanganj ', 'Kishangarh ', 'Kochi ', 'Kolar ', 'Kolhapur ', 'Kolkata ', 'Kollam ', 'Korba ', 'Kota ', 'Kozhikode ', 'Krishnanagar ', 'Kulti ', 'Kumbakonam ', 'Kurichi ', 'Kurnool ', 'Lakhimpur ', 'Lalitpur ', 'Latur ', 'Loni ', 'Lucknow ', 'Ludhiana ', 'Machilipatnam ', 'Madanapalle ', 'Madavaram ', 'Madhyamgram ', 'Madurai ', 'Mahbubnagar ', 'Mahesana ', 'Maheshtala ', 'Mainpuri ', 'Malegaon ', 'Malerkotla ', 'Mandoli ', 'Mandsaur ', 'Mandya ', 'Mangalore ', 'Mango ', 'Mathura ', 'Maunath Bhanjan ', 'Medinipur ', 'Meerut ', 'Mira Bhayander ', 'Miryalaguda ', 'Mirzapur-cum-Vindhyachal ', 'Modinagar ', 'Moga ', 'Moradabad ', 'Morena ', 'Morvi ', 'Motihari ', 'Mughalsarai ', 'Muktsar ', 'Munger ', 'Murwara ', 'Mustafabad ', 'Muzaffarnagar ', 'Muzaffarpur ', 'Mysore ', 'Nabadwip ', 'Nadiad ', 'Nagaon ', 'Nagapattinam ', 'Nagaur ', 'Nagda ', 'Nagercoil ', 'Nagpur ', 'Naihati ', 'Nalgonda ', 'Nanded Waghala ', 'Nandurbar ', 'Nandyal ', 'Nangloi Jat ', 'Narasaraopet ', 'Nashik ', 'Navi Mumbai ', 'Navi Mumbai Panvel Raigarh ', 'Navsari ', 'Neemuch ', 'Nellore ', 'New Delhi ', 'Neyveli ', 'Nizamabad ', 'Noida ', 'North Barrackpur ', 'North Dum Dum ', 'Ongole ', 'Orai ', 'Osmanabad ', 'Ozhukarai ', 'Palakkad ', 'Palanpur ', 'Pali ', 'Pallavaram ', 'Palwal ', 'Panchkula ', 'Panihati ', 'Panipat ', 'Panvel ', 'Parbhani ', 'Patan ', 'Pathankot ', 'Patiala ', 'Patna ', 'Pilibhit ', 'Pimpri Chinchwad ', 'Pithampur ', 'Porbandar ', 'Port Blair ', 'Proddatur ', 'Puducherry ', 'Pudukkottai ', 'Pune ', 'Puri ', 'Purnia ', 'Puruliya ', 'Rae Bareli ', 'Raichur ', 'Raiganj ', 'Raigarh ', 'Raipur ', 'Rajahmundry ', 'Rajapalayam ', 'Rajarhat Gopalpur ', 'Rajkot ', 'Rajnandgaon ', 'Rajpur Sonarpur ', 'Ramagundam ', 'Rampur ', 'Ranchi ', 'Ranibennur ', 'Raniganj ', 'Ratlam ', 'Raurkela Industrial Township ', 'Raurkela Town ', 'Rewa ', 'Rewari ', 'Rishra ', 'Robertson Pet ', 'Rohtak ', 'Roorkee ', 'Rudrapur ', 'S.A.S. Nagar ', 'Sagar ', 'Saharanpur ', 'Saharsa ', 'Salem ', 'Sambalpur ', 'Sambhal ', 'Sangli Miraj Kupwad ', 'Santipur ', 'Sasaram ', 'Satara ', 'Satna ', 'Sawai Madhopur ', 'Secunderabad ', 'Sehore ', 'Seoni ', 'Serampore ', 'Shahjahanpur ', 'Shamli ', 'Shikohabad ', 'Shillong ', 'Shimla ', 'Shimoga ', 'Shivpuri ', 'Sikar ', 'Silchar ', 'Siliguri ', 'Singrauli ', 'Sirsa ', 'Sitapur ', 'Siwan ', 'Solapur ', 'Sonipat ', 'South Dum Dum ', 'Srikakulam ', 'Srinagar ', 'Sujangarh ', 'Sultan Pur Majra ', 'Sultanpur ', 'Surat ', 'Surendranagar Dudhrej ', 'Suryapet ', 'Tadepalligudem ', 'Tadpatri ', 'Tambaram ', 'Tenali ', 'Thane ', 'Thanesar ', 'Thanjavur ', 'Thiruvananthapuram ', 'Thoothukkudi ', 'Thrissur ', 'Tiruchirappalli ', 'Tirunelveli ', 'Tirupati ', 'Tiruppur ', 'Tiruvannamalai ', 'Tiruvottiyur ', 'Titagarh ', 'Tonk ', 'Tumkur ', 'Udaipur ', 'Udgir ', 'Udupi ', 'Ujjain ', 'Ulhasnagar ', 'Uluberia ', 'Unnao ', 'Uttarpara Kotrung ', 'Vadodara ', 'Valsad ', 'Varanasi ', 'Vasai Virar City ', 'Vellore ', 'Veraval ', 'Vidisha ', 'Vijayawada ', 'Visakhapatnam', 'Vizianagaram ', 'Warangal ', 'Wardha ', 'Yamunanagar ', 'YAVATMAL ', 'ABOHAR ', 'ACHALPUR ', 'ADILABAD ', 'ADITYAPUR ', 'ADONI ', 'AGARTALA ', 'AGRA ', 'AHMADABAD ', 'AHMADNAGAR ', 'AIZAWL ', 'AJMER ', 'AKBARPUR ', 'AKOLA ', 'ALANDUR ', 'ALAPPUZHA ', 'ALIGARH ', 'ALLAHABAD ', 'ALWAR ', 'AMBALA ', 'AMBALA SADAR ', 'AMBARNATH ', 'AMBATTUR ', 'AMBIKAPUR ', 'AMBUR ', 'AMRAVATI ', 'AMRELI ', 'AMRITSAR ', 'AMROHA ', 'ANAND ', 'ANANTAPUR ', 'ANANTNAG ', 'ARRAH ', 'ASANSOL ', 'ASHOKNAGAR KALYANGARH ', 'AURANGABAD ', 'AURANGABAD ', 'AVADI ', 'AZAMGARH ', 'BADLAPUR ', 'BAGAHA ', 'BAGALKOT ', 'BAHADURGARH ', 'BAHARAMPUR ', 'BAHRAICH ', 'BAIDYABATI ', 'BALESHWAR TOWN ', 'BALLIA ', 'BALLY ', 'BALLY CITY', 'BALURGHAT ', 'BANDA ', 'BANKURA ', 'BANSBERIA ', 'BANSWARA ', 'BARAN ', 'BARANAGAR ', 'BARASAT ', 'BARAUT ', 'BARDDHAMAN ', 'BAREILLY ', 'BARIPADA TOWN ', 'BARNALA ', 'BARRACKPUR ', 'BARSHI ', 'BASIRHAT ', 'BASTI ', 'BATALA ', 'BATHINDA ', 'BEAWAR ', 'BEGUSARAI ', 'BELGAUM ', 'BELLARY ', 'BENGALURU', 'BETTIAH ', 'BETUL ', 'BHADRAK ', 'BHADRAVATI ', 'BHADRESWAR ', 'BHAGALPUR ', 'BHALSWA JAHANGIR PUR ', 'BHARATPUR ', 'BHARUCH ', 'BHATPARA ', 'BHAVNAGAR ', 'BHILAI NAGAR ', 'BHILWARA ', 'BHIMAVARAM ', 'BHIND ', 'BHIWADI ', 'BHIWANDI ', 'BHIWANI ', 'BHOPAL ', 'BHUBANESWAR TOWN ', 'BHUJ ', 'BHUSAWAL ', 'BID ', 'BIDAR ', 'BIDHAN NAGAR ', 'BIHARSHARIF ', 'BIJAPUR ', 'BIKANER ', 'BILASPUR ', 'BOKARO STEEL CITY ', 'BONGAON ', 'BOTAD ', 'BRAHMAPUR TOWN ', 'BUDAUN ', 'BULANDSHAHR ', 'BUNDI ', 'BURARI ', 'BURHANPUR ', 'BUXAR ', 'CHAMPDANI ', 'CHANDANNAGAR ', 'CHANDAUSI ', 'CHANDIGARH ', 'CHANDRAPUR ', 'CHAPRA ', 'CHAS ', 'CHENNAI ', 'CHHATTARPUR ', 'CHHINDWARA ', 'CHIKMAGALUR ', 'CHILAKALURIPET ', 'CHITRADURGA ', 'CHITTAURGARH ', 'CHITTOOR ', 'CHURU ', 'COIMBATORE ', 'CUDDALORE ', 'CUTTACK ', 'DABGRAM ', 'DALLO PURA ', 'DAMOH ', 'DARBHANGA ', 'DARJILING ', 'DATIA ', 'DAVANAGERE ', 'DEESA ', 'DEHRADUN ', 'DEHRI ', 'DELHI ', 'DELHI CANTONMENT ', 'DEOGHAR ', 'DEOLI ', 'DEORIA ', 'DEWAS ', 'DHANBAD ', 'DHARMAVARAM ', 'DHAULPUR ', 'DHULE ', 'DIBRUGARH ', 'DIMAPUR ', 'DINAPUR NIZAMAT ', 'DINDIGUL ', 'DUM DUM ', 'DURG ', 'DURGAPUR ', 'ELURU ', 'ENGLISH BAZAR ', 'ERODE ', 'ETAH ', 'ETAWAH ', 'FAIZABAD ', 'FARIDABAD ', 'FARRUKHABAD-CUM-FATEHGARH ', 'FATEHPUR ', 'FIROZABAD ', 'FIROZPUR ', 'GADAG-BETIGERI ', 'GANDHIDHAM ', 'GANDHINAGAR ', 'GANGANAGAR ', 'GANGAPUR CITY ', 'GANGAWATI ', 'GAYA ', 'GHAZIPUR ', 'GIRIDIH ', 'GODHRA ', 'GOKAL PUR ', 'GONDA ', 'GONDAL ', 'GONDIYA ', 'GORAKHPUR ', 'GREATER HYDERABAD ', 'GREATER MUMBAI ', 'GREATER NOIDA ', 'GUDIVADA ', 'GULBARGA ', 'GUNA ', 'GUNTAKAL ', 'GUNTUR ', 'GURGAON ', 'GUWAHATI ', 'GWALIOR ', 'HABRA ', 'HAJIPUR ', 'HALDIA ', 'HALDWANI-CUM-KATHGODAM ', 'HALISAHAR ', 'HANUMANGARH ', 'HAORA ', 'HAPUR ', 'HARDOI ', 'HARDWAR ', 'HASSAN ', 'HASTSAL ', 'HATHRAS ', 'HAZARIBAG ', 'HINDAUN ', 'HINDUPUR ', 'HINGANGHAT ', 'HISAR ', 'HOSHANGABAD ', 'HOSHIARPUR ', 'HOSPET ', 'HOSUR ', 'HUBLI-DHARWAD ', 'HUGLI-CHINSURAH ', 'ICHALKARANJI ', 'IMPHAL ', 'INDORE ', 'JABALPUR ', 'JAGADHRI ', 'JAGDALPUR ', 'JAIPUR ', 'JALANDHAR ', 'JALGAON ', 'JALNA ', 'JALPAIGURI ', 'JAMALPUR ', 'JAMMU ', 'JAMNAGAR ', 'JAMSHEDPUR ', 'JAMURIA ', 'JAUNPUR ', 'JEHANABAD ', 'JETPUR NAVAGADH ', 'JHANSI ', 'JHUNJHUNUN ', 'JIND ', 'JODHPUR ', 'JUNAGADH ', 'KADAPA ', 'KAITHAL ', 'KAKINADA ', 'KALOL ', 'KALYANI ', 'KAMARHATI ', 'KANCHEEPURAM ', 'KANCHRAPARA ', 'KANPUR ', 'KANPUR CITY', 'KARAIKKUDI ', 'KARAWAL NAGAR ', 'KARIMNAGAR ', 'KARNAL ', 'KASGANJ ', 'KASHIPUR ', 'KATIHAR ', 'KHAMMAM ', 'KHANDWA ', 'KHANNA ', 'KHARAGPUR ', 'KHARDAHA ', 'KHARGONE ', 'KHORA ', 'KHURJA ', 'KIRARI SULEMAN NAGAR ', 'KISHANGANJ ', 'KISHANGARH ', 'KOCHI ', 'KOLAR ', 'KOLHAPUR ', 'KOLKATA ', 'KOLLAM ', 'KORBA ', 'KOTA ', 'KOZHIKODE ', 'KRISHNANAGAR ', 'KULTI ', 'KUMBAKONAM ', 'KURICHI ', 'KURNOOL ', 'LAKHIMPUR ', 'LALITPUR ', 'LATUR ', 'LONI ', 'LUCKNOW ', 'LUDHIANA ', 'MACHILIPATNAM ', 'MADANAPALLE ', 'MADAVARAM ', 'MADHYAMGRAM ', 'MADURAI ', 'MAHBUBNAGAR ', 'MAHESANA ', 'MAHESHTALA ', 'MAINPURI ', 'MALEGAON ', 'MALERKOTLA ', 'MANDOLI ', 'MANDSAUR ', 'MANDYA ', 'MANGALORE ', 'MANGO ', 'MATHURA ', 'MAUNATH BHANJAN ', 'MEDINIPUR ', 'MEERUT ', 'MIRA BHAYANDER ', 'MIRYALAGUDA ', 'MIRZAPUR-CUM-VINDHYACHAL ', 'MODINAGAR ', 'MOGA ', 'MORADABAD ', 'MORENA ', 'MORVI ', 'MOTIHARI ', 'MUGHALSARAI ', 'MUKTSAR ', 'MUNGER ', 'MURWARA ', 'MUSTAFABAD ', 'MUZAFFARNAGAR ', 'MUZAFFARPUR ', 'MYSORE ', 'NABADWIP ', 'NADIAD ', 'NAGAON ', 'NAGAPATTINAM ', 'NAGAUR ', 'NAGDA ', 'NAGERCOIL ', 'NAGPUR ', 'NAIHATI ', 'NALGONDA ', 'NANDED WAGHALA ', 'NANDURBAR ', 'NANDYAL ', 'NANGLOI JAT ', 'NARASARAOPET ', 'NASHIK ', 'NAVI MUMBAI ', 'NAVI MUMBAI PANVEL RAIGARH ', 'NAVSARI ', 'NEEMUCH ', 'NELLORE ', 'NEW DELHI ', 'NEYVELI ', 'NIZAMABAD ', 'NOIDA ', 'NORTH BARRACKPUR ', 'NORTH DUM DUM ', 'ONGOLE ', 'ORAI ', 'OSMANABAD ', 'OZHUKARAI ', 'PALAKKAD ', 'PALANPUR ', 'PALI ', 'PALLAVARAM ', 'PALWAL ', 'PANCHKULA ', 'PANIHATI ', 'PANIPAT ', 'PANVEL ', 'PARBHANI ', 'PATAN ', 'PATHANKOT ', 'PATIALA ', 'PATNA ', 'PILIBHIT ', 'PIMPRI CHINCHWAD ', 'PITHAMPUR ', 'PORBANDAR ', 'PORT BLAIR ', 'PRODDATUR ', 'PUDUCHERRY ', 'PUDUKKOTTAI ', 'PUNE ', 'PURI ', 'PURNIA ', 'PURULIYA ', 'RAE BARELI ', 'RAICHUR ', 'RAIGANJ ', 'RAIGARH ', 'RAIPUR ', 'RAJAHMUNDRY ', 'RAJAPALAYAM ', 'RAJARHAT GOPALPUR ', 'RAJKOT ', 'RAJNANDGAON ', 'RAJPUR SONARPUR ', 'RAMAGUNDAM ', 'RAMPUR ', 'RANCHI ', 'RANIBENNUR ', 'RANIGANJ ', 'RATLAM ', 'RAURKELA INDUSTRIAL TOWNSHIP ', 'RAURKELA TOWN ', 'REWA ', 'REWARI ', 'RISHRA ', 'ROBERTSON PET ', 'ROHTAK ', 'ROORKEE ', 'RUDRAPUR ', 'S.A.S. NAGAR ', 'SAGAR ', 'SAHARANPUR ', 'SAHARSA ', 'SALEM ', 'SAMBALPUR ', 'SAMBHAL ', 'SANGLI MIRAJ KUPWAD ', 'SANTIPUR ', 'SASARAM ', 'SATARA ', 'SATNA ', 'SAWAI MADHOPUR ', 'SECUNDERABAD ', 'SEHORE ', 'SEONI ', 'SERAMPORE ', 'SHAHJAHANPUR ', 'SHAMLI ', 'SHIKOHABAD ', 'SHILLONG ', 'SHIMLA ', 'SHIMOGA ', 'SHIVPURI ', 'SIKAR ', 'SILCHAR ', 'SILIGURI ', 'SINGRAULI ', 'SIRSA ', 'SITAPUR ', 'SIWAN ', 'SOLAPUR ', 'SONIPAT ', 'SOUTH DUM DUM ', 'SRIKAKULAM ', 'SRINAGAR ', 'SUJANGARH ', 'SULTAN PUR MAJRA ', 'SULTANPUR ', 'SURAT ', 'SURENDRANAGAR DUDHREJ ', 'SURYAPET ', 'TADEPALLIGUDEM ', 'TADPATRI ', 'TAMBARAM ', 'TENALI ', 'THANE ', 'THANESAR ', 'THANJAVUR ', 'THIRUVANANTHAPURAM ', 'THOOTHUKKUDI ', 'THRISSUR ', 'TIRUCHIRAPPALLI ', 'TIRUNELVELI ', 'TIRUPATI ', 'TIRUPPUR ', 'TIRUVANNAMALAI ', 'TIRUVOTTIYUR ', 'TITAGARH ', 'TONK ', 'TUMKUR ', 'UDAIPUR ', 'UDGIR ', 'UDUPI ', 'UJJAIN ', 'ULHASNAGAR ', 'ULUBERIA ', 'UNNAO ', 'UTTARPARA KOTRUNG ', 'VADODARA ', 'VALSAD ', 'VARANASI ', 'VASAI VIRAR CITY ', 'VELLORE ', 'VERAVAL ', 'VIDISHA ', 'VIJAYAWADA ', 'VISAKHAPATNAM', 'VIZIANAGARAM ', 'WARANGAL ', 'WARDHA ', 'YAMUNANAGAR ', 'YAVATMAL ', 'YAVATMAL ', 'aBOHAR ', 'aCHALPUR ', 'aDILABAD ', 'aDITYAPUR ', 'aDONI ', 'aGARTALA ', 'aGRA ', 'aHMADABAD ', 'aHMADNAGAR ', 'aIZAWL ', 'aJMER ', 'aKBARPUR ', 'aKOLA ', 'aLANDUR ', 'aLAPPUZHA ', 'aLIGARH ', 'aLLAHABAD ', 'aLWAR ', 'aMBALA ', 'aMBALA sADAR ', 'aMBARNATH ', 'aMBATTUR ', 'aMBIKAPUR ', 'aMBUR ', 'aMRAVATI ', 'aMRELI ', 'aMRITSAR ', 'aMROHA ', 'aNAND ', 'aNANTAPUR ', 'aNANTNAG ', 'aRRAH ', 'aSANSOL ', 'aSHOKNAGAR kALYANGARH ', 'aURANGABAD ', 'aURANGABAD ', 'aVADI ', 'aZAMGARH ', 'bADLAPUR ', 'bAGAHA ', 'bAGALKOT ', 'bAHADURGARH ', 'bAHARAMPUR ', 'bAHRAICH ', 'bAIDYABATI ', 'bALESHWAR tOWN ', 'bALLIA ', 'bALLY ', 'bALLY cITY', 'bALURGHAT ', 'bANDA ', 'bANKURA ', 'bANSBERIA ', 'bANSWARA ', 'bARAN ', 'bARANAGAR ', 'bARASAT ', 'bARAUT ', 'bARDDHAMAN ', 'bAREILLY ', 'bARIPADA tOWN ', 'bARNALA ', 'bARRACKPUR ', 'bARSHI ', 'bASIRHAT ', 'bASTI ', 'bATALA ', 'bATHINDA ', 'bEAWAR ', 'bEGUSARAI ', 'bELGAUM ', 'bELLARY ', 'bENGALURU', 'bETTIAH ', 'bETUL ', 'bHADRAK ', 'bHADRAVATI ', 'bHADRESWAR ', 'bHAGALPUR ', 'bHALSWA jAHANGIR pUR ', 'bHARATPUR ', 'bHARUCH ', 'bHATPARA ', 'bHAVNAGAR ', 'bHILAI nAGAR ', 'bHILWARA ', 'bHIMAVARAM ', 'bHIND ', 'bHIWADI ', 'bHIWANDI ', 'bHIWANI ', 'bHOPAL ', 'bHUBANESWAR tOWN ', 'bHUJ ', 'bHUSAWAL ', 'bID ', 'bIDAR ', 'bIDHAN nAGAR ', 'bIHARSHARIF ', 'bIJAPUR ', 'bIKANER ', 'bILASPUR ', 'bOKARO sTEEL cITY ', 'bONGAON ', 'bOTAD ', 'bRAHMAPUR tOWN ', 'bUDAUN ', 'bULANDSHAHR ', 'bUNDI ', 'bURARI ', 'bURHANPUR ', 'bUXAR ', 'cHAMPDANI ', 'cHANDANNAGAR ', 'cHANDAUSI ', 'cHANDIGARH ', 'cHANDRAPUR ', 'cHAPRA ', 'cHAS ', 'cHENNAI ', 'cHHATTARPUR ', 'cHHINDWARA ', 'cHIKMAGALUR ', 'cHILAKALURIPET ', 'cHITRADURGA ', 'cHITTAURGARH ', 'cHITTOOR ', 'cHURU ', 'cOIMBATORE ', 'cUDDALORE ', 'cUTTACK ', 'dABGRAM ', 'dALLO pURA ', 'dAMOH ', 'dARBHANGA ', 'dARJILING ', 'dATIA ', 'dAVANAGERE ', 'dEESA ', 'dEHRADUN ', 'dEHRI ', 'dELHI ', 'dELHI cANTONMENT ', 'dEOGHAR ', 'dEOLI ', 'dEORIA ', 'dEWAS ', 'dHANBAD ', 'dHARMAVARAM ', 'dHAULPUR ', 'dHULE ', 'dIBRUGARH ', 'dIMAPUR ', 'dINAPUR nIZAMAT ', 'dINDIGUL ', 'dUM dUM ', 'dURG ', 'dURGAPUR ', 'eLURU ', 'eNGLISH bAZAR ', 'eRODE ', 'eTAH ', 'eTAWAH ', 'fAIZABAD ', 'fARIDABAD ', 'fARRUKHABAD-cUM-fATEHGARH ', 'fATEHPUR ', 'fIROZABAD ', 'fIROZPUR ', 'gADAG-bETIGERI ', 'gANDHIDHAM ', 'gANDHINAGAR ', 'gANGANAGAR ', 'gANGAPUR cITY ', 'gANGAWATI ', 'gAYA ', 'gHAZIPUR ', 'gIRIDIH ', 'gODHRA ', 'gOKAL pUR ', 'gONDA ', 'gONDAL ', 'gONDIYA ', 'gORAKHPUR ', 'gREATER hYDERABAD ', 'gREATER mUMBAI ', 'gREATER nOIDA ', 'gUDIVADA ', 'gULBARGA ', 'gUNA ', 'gUNTAKAL ', 'gUNTUR ', 'gURGAON ', 'gUWAHATI ', 'gWALIOR ', 'hABRA ', 'hAJIPUR ', 'hALDIA ', 'hALDWANI-cUM-kATHGODAM ', 'hALISAHAR ', 'hANUMANGARH ', 'hAORA ', 'hAPUR ', 'hARDOI ', 'hARDWAR ', 'hASSAN ', 'hASTSAL ', 'hATHRAS ', 'hAZARIBAG ', 'hINDAUN ', 'hINDUPUR ', 'hINGANGHAT ', 'hISAR ', 'hOSHANGABAD ', 'hOSHIARPUR ', 'hOSPET ', 'hOSUR ', 'hUBLI-dHARWAD ', 'hUGLI-cHINSURAH ', 'iCHALKARANJI ', 'iMPHAL ', 'iNDORE ', 'jABALPUR ', 'jAGADHRI ', 'jAGDALPUR ', 'jAIPUR ', 'jALANDHAR ', 'jALGAON ', 'jALNA ', 'jALPAIGURI ', 'jAMALPUR ', 'jAMMU ', 'jAMNAGAR ', 'jAMSHEDPUR ', 'jAMURIA ', 'jAUNPUR ', 'jEHANABAD ', 'jETPUR nAVAGADH ', 'jHANSI ', 'jHUNJHUNUN ', 'jIND ', 'jODHPUR ', 'jUNAGADH ', 'kADAPA ', 'kAITHAL ', 'kAKINADA ', 'kALOL ', 'kALYANI ', 'kAMARHATI ', 'kANCHEEPURAM ', 'kANCHRAPARA ', 'kANPUR ', 'kANPUR cITY', 'kARAIKKUDI ', 'kARAWAL nAGAR ', 'kARIMNAGAR ', 'kARNAL ', 'kASGANJ ', 'kASHIPUR ', 'kATIHAR ', 'kHAMMAM ', 'kHANDWA ', 'kHANNA ', 'kHARAGPUR ', 'kHARDAHA ', 'kHARGONE ', 'kHORA ', 'kHURJA ', 'kIRARI sULEMAN nAGAR ', 'kISHANGANJ ', 'kISHANGARH ', 'kOCHI ', 'kOLAR ', 'kOLHAPUR ', 'kOLKATA ', 'kOLLAM ', 'kORBA ', 'kOTA ', 'kOZHIKODE ', 'kRISHNANAGAR ', 'kULTI ', 'kUMBAKONAM ', 'kURICHI ', 'kURNOOL ', 'lAKHIMPUR ', 'lALITPUR ', 'lATUR ', 'lONI ', 'lUCKNOW ', 'lUDHIANA ', 'mACHILIPATNAM ', 'mADANAPALLE ', 'mADAVARAM ', 'mADHYAMGRAM ', 'mADURAI ', 'mAHBUBNAGAR ', 'mAHESANA ', 'mAHESHTALA ', 'mAINPURI ', 'mALEGAON ', 'mALERKOTLA ', 'mANDOLI ', 'mANDSAUR ', 'mANDYA ', 'mANGALORE ', 'mANGO ', 'mATHURA ', 'mAUNATH bHANJAN ', 'mEDINIPUR ', 'mEERUT ', 'mIRA bHAYANDER ', 'mIRYALAGUDA ', 'mIRZAPUR-cUM-vINDHYACHAL ', 'mODINAGAR ', 'mOGA ', 'mORADABAD ', 'mORENA ', 'mORVI ', 'mOTIHARI ', 'mUGHALSARAI ', 'mUKTSAR ', 'mUNGER ', 'mURWARA ', 'mUSTAFABAD ', 'mUZAFFARNAGAR ', 'mUZAFFARPUR ', 'mYSORE ', 'nABADWIP ', 'nADIAD ', 'nAGAON ', 'nAGAPATTINAM ', 'nAGAUR ', 'nAGDA ', 'nAGERCOIL ', 'nAGPUR ', 'nAIHATI ', 'nALGONDA ', 'nANDED wAGHALA ', 'nANDURBAR ', 'nANDYAL ', 'nANGLOI jAT ', 'nARASARAOPET ', 'nASHIK ', 'nAVI mUMBAI ', 'nAVI mUMBAI pANVEL rAIGARH ', 'nAVSARI ', 'nEEMUCH ', 'nELLORE ', 'nEW dELHI ', 'nEYVELI ', 'nIZAMABAD ', 'nOIDA ', 'nORTH bARRACKPUR ', 'nORTH dUM dUM ', 'oNGOLE ', 'oRAI ', 'oSMANABAD ', 'oZHUKARAI ', 'pALAKKAD ', 'pALANPUR ', 'pALI ', 'pALLAVARAM ', 'pALWAL ', 'pANCHKULA ', 'pANIHATI ', 'pANIPAT ', 'pANVEL ', 'pARBHANI ', 'pATAN ', 'pATHANKOT ', 'pATIALA ', 'pATNA ', 'pILIBHIT ', 'pIMPRI cHINCHWAD ', 'pITHAMPUR ', 'pORBANDAR ', 'pORT bLAIR ', 'pRODDATUR ', 'pUDUCHERRY ', 'pUDUKKOTTAI ', 'pUNE ', 'pURI ', 'pURNIA ', 'pURULIYA ', 'rAE bARELI ', 'rAICHUR ', 'rAIGANJ ', 'rAIGARH ', 'rAIPUR ', 'rAJAHMUNDRY ', 'rAJAPALAYAM ', 'rAJARHAT gOPALPUR ', 'rAJKOT ', 'rAJNANDGAON ', 'rAJPUR sONARPUR ', 'rAMAGUNDAM ', 'rAMPUR ', 'rANCHI ', 'rANIBENNUR ', 'rANIGANJ ', 'rATLAM ', 'rAURKELA iNDUSTRIAL tOWNSHIP ', 'rAURKELA tOWN ', 'rEWA ', 'rEWARI ', 'rISHRA ', 'rOBERTSON pET ', 'rOHTAK ', 'rOORKEE ', 'rUDRAPUR ', 's.a.s. nAGAR ', 'sAGAR ', 'sAHARANPUR ', 'sAHARSA ', 'sALEM ', 'sAMBALPUR ', 'sAMBHAL ', 'sANGLI mIRAJ kUPWAD ', 'sANTIPUR ', 'sASARAM ', 'sATARA ', 'sATNA ', 'sAWAI mADHOPUR ', 'sECUNDERABAD ', 'sEHORE ', 'sEONI ', 'sERAMPORE ', 'sHAHJAHANPUR ', 'sHAMLI ', 'sHIKOHABAD ', 'sHILLONG ', 'sHIMLA ', 'sHIMOGA ', 'sHIVPURI ', 'sIKAR ', 'sILCHAR ', 'sILIGURI ', 'sINGRAULI ', 'sIRSA ', 'sITAPUR ', 'sIWAN ', 'sOLAPUR ', 'sONIPAT ', 'sOUTH dUM dUM ', 'sRIKAKULAM ', 'sRINAGAR ', 'sUJANGARH ', 'sULTAN pUR mAJRA ', 'sULTANPUR ', 'sURAT ', 'sURENDRANAGAR dUDHREJ ', 'sURYAPET ', 'tADEPALLIGUDEM ', 'tADPATRI ', 'tAMBARAM ', 'tENALI ', 'tHANE ', 'tHANESAR ', 'tHANJAVUR ', 'tHIRUVANANTHAPURAM ', 'tHOOTHUKKUDI ', 'tHRISSUR ', 'tIRUCHIRAPPALLI ', 'tIRUNELVELI ', 'tIRUPATI ', 'tIRUPPUR ', 'tIRUVANNAMALAI ', 'tIRUVOTTIYUR ', 'tITAGARH ', 'tONK ', 'tUMKUR ', 'uDAIPUR ', 'uDGIR ', 'uDUPI ', 'uJJAIN ', 'uLHASNAGAR ', 'uLUBERIA ', 'uNNAO ', 'uTTARPARA kOTRUNG ', 'vADODARA ', 'vALSAD ', 'vARANASI ', 'vASAI vIRAR cITY ', 'vELLORE ', 'vERAVAL ', 'vIDISHA ', 'vIJAYAWADA ', 'vISAKHAPATNAM', 'vIZIANAGARAM ', 'wARANGAL ', 'wARDHA ', 'yAMUNANAGAR ', 'yAVATMAL ']
    #print(lineList)
    def common(a,b):
        c = [value for value in a if value in b]
        return c

    print("Executing")

    #print(stopwords)

    address = []
    names = []
    for i in range(0, len(sentence)):

        words = word_tokenize(sentence[i])
        #print(words)
        e = common(words, lineList)
        #print(e)
        d = common(words, chr)
        f = common(words, shop_idt)
        #print(f)
        #print("YOYO")
        #print(shop_idt)
        g = common(words, stopwords)
        #print(g)
        # if e != []:
        #     # if d != []:
        #     #     address.append(sentence[i])
        #     # else:
        #     #     # if no comma but still place, check if pincode is a part of the string
        #     #     # pins=re.findall(r"[0-9]{6} |[0-9]{3}\s[0-9]{3}",sentence[i])
        #     #     # if pins != []:
        #     if (i>1):
        #         address.append(sentence[i-2])
        #         address.append(sentence[i-1])
        #         address.append(sentence[i])
        #         address.append(sentence[i+1 ])
        #
        #     if f != []:
        #         names.append(sentence[i])
        #
        # else:
        #     if d != []:
        #         address.append(sentence[i])
        #     else:
        #         if f!=[]:
        #             names.append(sentence[i])
        #         elif g!=[]:
        #             stopwords.append(sentence[i])

        if e !=[]:
            if d!=[]:
                address.append(sentence[i])
                #address.append(sentence[i - 2])
                address.append(sentence[i-1])
                #address.append(sentence[i])
                address.append(sentence[i+1])
            else:
                names.append(sentence[i])

        else:
            if d != []:
                address.append(sentence[i])
            else:
                if f!=[]:
                    names.append(sentence[i])
                elif g!=[]:
                    stopwords.append(sentence[i])

    print("Address is")
    print(address)
    print("Name is")

    print(names)
    #for i in address:
        #stopwords.append(i)

    # for i in len(stopwords):
    #     print(stopwords[i])
    #     print(\n)

    for s in range(0, len(stopwords)):
        if stopwords[s] == ',':
            stopwords[s] = '@'

    #x aprint(stopwords)
    with open('dataset.txt', 'w') as f:
        for i in range(0, len(stopwords)):
            f.write("%s\n" % (stopwords[i]))