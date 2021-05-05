#################################################################
# Title: Extract 10-K Sentences
# Author: Sam Cohen

# Notes:
# These functions encompass downloading 10-K filings, extracting sentences from
# Item 1A., and outputting those 10-K sentences and basic information about each
# company to text files to be used in later analyses.
##################################################################

# Packages
import os
from sec_edgar_downloader import Downloader
from datapackage import Package
import nltk
import pandas as pd
from bs4 import BeautifulSoup
import urllib
import re, requests
from multiprocessing import Process, Manager

#nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

def companies_sp_500():
    """
    Import a list of all publicly traded companies in the S&P 500 and some characteristics about each one.
    Code comes from: https://datahub.io/core/s-and-p-500-companies#python
    :return: the list of all companies in the S&P 500
    """
    package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')

    # print list of all resources:
    #print(package.resource_names)

    sp500 = []
    # print processed tabular data (if exists any)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            sp500 = resource.read()

    return sp500

def getCIKs(TICKERS):
    """
    This code gets the Central Index Key (CIK) associated with a stock ticker.
    This code comes from here: https://gist.github.com/dougvk/8499335
    :param TICKERS:
    :return: A dictionary of stock ticker - CIK value
    """
    URL = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
    CIK_RE = re.compile(r'.*CIK=(\d{10}).*')
    cik_dict = {}
    for ticker in TICKERS:
        f = requests.get(URL.format(ticker[0]), stream = True)
        results = CIK_RE.findall(f.text)
        if len(results):
            results[0] = int(re.sub('\.[0]*', '.', results[0]))
            cik_dict[str(ticker).upper()] = str(results[0])
    f = open('cik_dict', 'w')
    f.close()
    print(cik_dict)
    return (cik_dict)

def read_file(filename):
    """
    Reads a text file.
    :param filename: name of the text file
    :return: a text file in Python
    """
    input_file_text = open(filename, encoding='utf-8').read()
    return input_file_text

def get_cik_translator():
    """
    This function utilizes a list of stock ticker - CIK found at the URL below on the SEC website.
    A dictionary that crosswalks stock ticker to CIK is output.
    :return: dictionary that crosswalks stock ticker to CIK
    """
    url = "https://www.sec.gov/include/ticker.txt"
    file = urllib.request.urlopen(url)#.read().decode('utf-8')
    crosswalk_dict = {}
    for line in file:
        decoded_line = line.decode("utf-8").split()
        crosswalk_dict[decoded_line[0]] = decoded_line[1]
    return crosswalk_dict

def convert_ticker_to_cik(ticker, crosswalk):
    """
    This helper function converts a ticker to CIK using the crosswalk produced by the get_cik_translator function.
    :param ticker: singular stock ticker
    :param crosswalk: the stock ticker - CIK dictionary
    :return: the CIK associated with a stock ticker
    """
    tick = ticker[0]
    tick = tick.lower()
    tick = tick.replace('.', '-')
    cik = crosswalk[tick]
    return cik

def get_key(val, dict):
    """
    Retrieve a key from a dictionary.
    Function comes from: https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    :param val:
    :param dict:
    :return:
    """
    for key, value in dict.items():
        if val == value:
            return key

    return "key doesn't exist"

def convert_cik_to_ticker(cik, crosswalk, sp500):
    """
    Converts CIK to stock ticker in a list of companies in the S&P 500.
    :param cik: Central Index Key
    :param crosswalk: dictionary of stock ticker - CIK
    :param sp500: list of companies in the S&P 500
    :return: stock ticker
    """
    ticker = get_key(cik, crosswalk)
    for i in sp500:
        if i[0].lower().replace('.', '-') == ticker:
            return i
    return 'DoesntWork'

def download_and_parse(actual_stock, ciks, dict):
    """
    This function is the meat and potatoes of downloading the SEC 10-K filings.
    It uses the sec_edgar_downloader package to download the 10-K filing.
    Then it uses code from https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2 to parse the 10-K filing for Item 1A.
    The code separates Item 1A into sentences and outputs it to a dictionary associated with the CIK value.
    :param actual_stock: the stock ticker
    :param ciks: CIK - stock ticker dictionary/crosswalk
    :param dict: a dictionary to store the 10-K Item 1A sentences
    :return: nothing, but it constantly updates/adds to the dictionary
    """
    if actual_stock[0] in ['BF.B', 'BF-B', 'bf-b', 'bf.b', 'HES', 'hes', 'hpe', 'HPE']:
        print("This stock has no CIK... issue there so I am skipping")
        return

    cik = convert_ticker_to_cik(actual_stock, ciks)
    cik = cik.zfill(10)

    dl = Downloader()
    #stock_ticker = "0001067983"
    dl.get("10-K", cik, after="2015-01-01", download_details=False)
    count = 0
    for root, dirs, files in os.walk("./sec-edgar-filings/{}/10-K/".format(cik)):

        # search through each years' 10-k filing
        for file in files:
            # find the txt document of the 10-K filing
            if file == 'full-submission.txt':

                try:
                    year = re.findall(r'\-[0-9][0-9]\-', root)
                    year = year[len(year) - 1][1:-1]

                    # certain stocks have issues for certain years. I will include code to exclude them here
                    # if year == 21 and stock_ticker in ("'APA'", "'ADM'"):
                    #     print("Skipping year {} for ticker {} due to issues...".format(year, stock_ticker))

                    # read the file
                    filing_text = read_file(root + '/' + file)

                    # this code comes from https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2

                    doc_start_pattern = re.compile(r'<DOCUMENT>')
                    doc_end_pattern = re.compile(r'</DOCUMENT>')
                    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
                    type_pattern = re.compile(r'<TYPE>[^\n]+')

                    # Create 3 lists with the span idices for each regex

                    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
                    ### First filter will give us document tag start <end> and document tag end's <start>
                    ### We will use this to later grab content in between these tags
                    doc_start_is = [x.end() for x in doc_start_pattern.finditer(filing_text)]
                    doc_end_is = [x.start() for x in doc_end_pattern.finditer(filing_text)]

                    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
                    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
                    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K'
                    ### as section names
                    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(filing_text)]

                    document = {}
                    # Create a loop to go through each section type and save only the 10-K section in the dictionary
                    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
                        if doc_type == '10-K':
                            document[doc_type] = filing_text[doc_start:doc_end]

                    regex = re.compile(r'(>(Item|ITEM)(\s|&#160;|&nbsp;|&#xa0;)(1A|1B)\.{0,1})|(ITEM\s(1A|1B))|(<B>Item</B><B></B><B>&nbsp;1A</B>)|(&nbsp;Item&nbsp;1B.)|(Item<font style="font-family:Times New Roman, Times, serif;font-size:10pt;">&nbsp;1B)|(Item<font style="font-family: Times New Roman, Times, serif; font-size: 10pt;">&nbsp;1B)')

                    matches = regex.finditer(document['10-K'])

                    test_df = pd.DataFrame([(x.group(), x.span()[0], x.span()[1]) for x in matches])

                    if len(test_df)==0:
                        print("error... didn't pick up anything")
                        break

                    test_df.columns = ['item', 'start', 'end']
                    test_df['item'] = test_df.item.str.lower()

                    test_df.replace('<font style="font-family:times new roman, times, serif;font-size:10pt;">', ' ', regex=True, inplace=True)
                    test_df.replace('<font style="font-family: times new roman, times, serif; font-size: 10pt;">', ' ',
                                    regex=True, inplace=True)
                    test_df.replace('&#160;', ' ', regex=True, inplace=True)
                    test_df.replace('&nbsp;', ' ', regex=True, inplace=True)
                    test_df.replace('&#xa0;', ' ', regex=True, inplace=True)
                    test_df.replace(' ', '', regex=True, inplace=True)
                    test_df.replace('\.', '', regex=True, inplace=True)
                    test_df.replace('>', '', regex=True, inplace=True)
                    test_df.replace('<b', '', regex=True, inplace=True)
                    test_df.replace('</b', '', regex=True, inplace=True)

                    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

                    pos_dat.set_index('item', inplace=True)

                    # Check conditionals here
                    if 'item1a' in pos_dat.index and 'item1b' in pos_dat.index:

                        item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item1b']]
                        item_1a_content = BeautifulSoup(item_1a_raw, 'lxml')

                        test_df["text"] = item_1a_content.get_text()
                        test_df.replace('([0-9]|[0-9][0-9])(\s{0,3})Table of Contents', ' ', regex=True, inplace=True)
                        test_df.replace('Table of Contents', ' ', regex=True, inplace=True)
                        test_df.replace('\s\s', ' ', regex=True, inplace=True)
                        test_df.replace('\\u200b', ' ', regex=True, inplace=True)
                        test_df.replace('\\n[0-9]', ' ', regex=True, inplace=True)
                        test_df.replace('[0-9]\\n', ' ', regex=True, inplace=True)
                        test_df.replace('\\xa0', ' ', regex=True, inplace=True)
                        test_df.replace('\\x92', ' ', regex=True, inplace=True)
                        test_df.replace('\\x93', ' ', regex=True, inplace=True)
                        test_df.replace('\\x94', ' ', regex=True, inplace=True)
                        test_df.replace('\\x95', ' ', regex=True, inplace=True)
                        test_df.replace('\\n', ' ', regex=True, inplace=True)
                        test_df.replace('\n', ' ', regex=False, inplace=True)

                        # output the text to the dict

                        sentences = nltk.sent_tokenize(str(test_df['text'][0]))

                        if count==0:
                            output_frame = pd.DataFrame([[year, sentences]], columns=["year", "text"])
                        else:
                            output_frame = output_frame.append(pd.DataFrame([[year, sentences]], columns=["year", "text"]), ignore_index=True)

                        dict[cik] = output_frame
                        print("finished processing ticker {} ({}) and added to dictionary for year {}".format(cik, actual_stock[0], year))
                        print(75*'')
                        count += 1

                    else:
                        regex = re.compile(
                            r'(>(Item|ITEM)(\s|&#160;|&nbsp;|&#xa0;)(1A|2)\.{0,1})|(ITEM\s(1A|2))|(<B>Item</B><B></B><B>&nbsp;1A</B>)|(&nbsp;Item&nbsp;2.)')

                        matches = regex.finditer(document['10-K'])

                        test_df = pd.DataFrame([(x.group(), x.span()[0], x.span()[1]) for x in matches])

                        if len(test_df) == 0:
                            print("error... didn't pick up anything")
                            break

                        test_df.columns = ['item', 'start', 'end']
                        test_df['item'] = test_df.item.str.lower()

                        test_df.replace('&#160;', ' ', regex=True, inplace=True)
                        test_df.replace('&nbsp;', ' ', regex=True, inplace=True)
                        test_df.replace('&#xa0;', ' ', regex=True, inplace=True)
                        test_df.replace(' ', '', regex=True, inplace=True)
                        test_df.replace('\.', '', regex=True, inplace=True)
                        test_df.replace('>', '', regex=True, inplace=True)
                        test_df.replace('<b', '', regex=True, inplace=True)
                        test_df.replace('</b', '', regex=True, inplace=True)

                        pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

                        pos_dat.set_index('item', inplace=True)

                        item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item2']]
                        item_1a_content = BeautifulSoup(item_1a_raw, 'lxml')

                        test_df["text"] = item_1a_content.get_text()
                        test_df.replace('([0-9]|[0-9][0-9])(\s{0,3})Table of Contents', ' ', regex=True, inplace=True)
                        test_df.replace('Table of Contents', ' ', regex=True, inplace=True)
                        test_df.replace('\s\s', ' ', regex=True, inplace=True)
                        test_df.replace('\\u200b', ' ', regex=True, inplace=True)
                        test_df.replace('\\n[0-9]', ' ', regex=True, inplace=True)
                        test_df.replace('[0-9]\\n', ' ', regex=True, inplace=True)
                        test_df.replace('\\xa0', ' ', regex=True, inplace=True)
                        test_df.replace('\\x92', ' ', regex=True, inplace=True)
                        test_df.replace('\\x93', ' ', regex=True, inplace=True)
                        test_df.replace('\\x94', ' ', regex=True, inplace=True)
                        test_df.replace('\\x95', ' ', regex=True, inplace=True)
                        test_df.replace('\\n', ' ', regex=True, inplace=True)
                        test_df.replace('\n', ' ', regex=False, inplace=True)

                        # output the text to the dict

                        sentences = nltk.sent_tokenize(str(test_df['text'][0]))

                        if count == 0:
                            output_frame = pd.DataFrame([[year, sentences]], columns=["year", "text"])
                        else:
                            output_frame = output_frame.append(pd.DataFrame([[year, sentences]], columns=["year", "text"]),
                                                               ignore_index=True)

                        dict[cik] = output_frame
                        print("finished processing ticker {} ({}) and added to dictionary for year {}".format(cik,
                                                                                                              actual_stock[0],
                                                                                                              year))
                        print(75 * '')
                        count += 1

                except:
                    print("error occurred")

def ten_k_dictionary(sp500, ciks):
    """
    This function calls the download_and_parse function for each ticker in the sp500 list and creates the dictionary where the 10-K sentences are stored.
    I attempted a parallel implementation (with code from https://stackoverflow.com/questions/38393269/fill-up-a-dictionary-in-parallel-with-multiprocessing), but failed.
    I leave this code commented out for future tinkering efforts.
    :param sp500: list of all companies in the S&P 500
    :param ciks: CIK - stock ticker crosswalk
    :return: a dictionary of all the 10-K sentences
    """
    # manager = Manager()
    # ten_k_dict = manager.dict()
    sp500.reverse()
    ten_k_dict = {}
    # with cf.ThreadPoolExecutor(max_workers=4) as executor:
    #     dict = executor.map(extract_10_k.ten_k_dictionary, sp500[77:], ciks)
    # job = [Process(target=download_and_parse, args=(ticker, ciks, ten_k_dict)) for ticker in sp500]
    # _ = [p.start() for p in job]
    # _ = [p.join() for p in job]
    for ticker in sp500:
        download_and_parse(ticker, ciks, ten_k_dict)

    return ten_k_dict

def write_to_txt(dict, cik_crosswalk, sp500):
    """
    This function writes the dictionary with the 10-K sentences to a text file.
    It also writes the supplementary information (ex. industry type) about each company to a separate text file for joint analyses later.
    :param dict: dictionary with 10-K climate sentences
    :param cik_crosswalk: CIK - stock ticker crosswalk
    :param sp500: list of companies in S&P 500
    :return: nothing, but outputs two text files - one with predictions, another with company information
    """
    ten_k_sents = open("Domain-Agnostic-Sentence-Specificity-Prediction/dataset/data/ten_k_sentences.txt", "w")
    ten_k_info = open("Domain-Agnostic-Sentence-Specificity-Prediction/dataset/data/ten_k_info.txt", "w")

    # write a first sentence as a dud because the specificity prediction algorithm cuts off the first obs
    ten_k_sents.writelines("first sentence - to be deleted" + '\n')
    ten_k_info.writelines("first sentence - to be deleted" + '\n')

    for key, val in dict.items():
        for i in range(len(val["text"])):
            for j in range(len(val["text"][i])):
                year = "20" + val["year"][i]
                # Need to do this step to remove the leading zeros, as I added them in a later process and
                # there are not leading zeros in the initial dictionary
                key_transformed = key.lstrip('0')
                sector = convert_cik_to_ticker(key_transformed, cik_crosswalk, sp500)

                # data structure is CIK, Year, Stock Ticker, Industry
                info = key + "\t" + year + '\t' + sector[0] + '\t' + sector[1] + '\t' + sector[2]

                ten_k_sents.writelines(val["text"][i][j] + '\n')
                ten_k_info.writelines(info + '\n')

def count_number_of_companies():
    """
    This function counts the number of companies and 10-Ks used.
    It is strictly used for analysis and awareness.
    :return: nothing, but prints the number of companies and 10-K filings.
    """
    count_company = 0
    count_10ks = 0
    for root, dirs, files in os.walk("./sec-edgar-filings/"):
        if root == "./sec-edgar-filings/":
            count_company = len(dirs)
        for _ in files:
            count_10ks += 1

    print("The number of companies included is:", count_company)
    print("The number of 10-K filings included is:", count_10ks)
