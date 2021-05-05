# climate-risk-capstone

This is the GitHub repository associated with my capstone project for the GWU Data Science Master's Program.

My capstone project is title "**Can You Clarify?: Using NLP to Determine the Specificity of Climate-Related Financial Disclosures**".

My work relies heavily on two papers. Their citations are listed here:

    * Kölbel, Julian and Leippold, Markus and Rillaerts, Jordy and Wang, Qian, Ask BERT: How Regulatory Disclosure of Transition and Physical Climate Risks affects the CDS Term Structure (June 1, 2020). Swiss Finance Institute Research Paper No. 21-19, Available at SSRN: https://ssrn.com/abstract=3616324 or http://dx.doi.org/10.2139/ssrn.3616324
    * Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019. https://arxiv.org/pdf/1811.05085.pdf
    
Much of the design of the BERT approach and the training dataset construction used for to fine-tune the BERT model came from the Kölbel et al. (2020) paper.
The modeling approach used for the sentence specificity prediction came from Ko et al.'s (2019) paper and GitHub repository (https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction).

## How to Run This Code
This code is a bit complex to run, as there are many moving parts and dependencies. When run fully, it takes ~3-5 hours to complete. Here is how to run it.

First, I only ran everything on a GPU. I ran on an AWS EC2 instance. It may be possible to run without a GPU/CUDA, but you would have to
follow the instructions on Ko et al.'s GitHub repository for the sentence prediction piece.

Next, to run the sentence prediction piece, you need to download a glove vector file (840B.300d). It needs to 
be downloaded inside the Domain-Agnostic-Sentence-Specificity-Prediciton folder. The commands to download it are below.

    wget https://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip

Next, you need to make sure you have all the requirements in terms of packages to run the file. I have created a requirements.txt
file that contains all these requirements. To install all the requirements, type the below command:

    pip install -r requirements.txt

or

    conda install -r requirements.txt

Next, you are ready to run the file! To run the entire file and see the output in the console, you can run the main.py file.

    python3 main.py
    
If you do not want your console getting clogged up, you can run the run.py file. This file runs the main.py, but outputs a log.txt
file with the contents of the output. That way you can run the file, leave, and come back a few hours later to check your log file.

One major thing to know is that the bulk of the data comes from the 10-K file downloads. These files get downloaded to a folder on your
computer and parsed. Make sure you have plenty of space (at least 10GBs) on your computer to handle the downloaded 10-K files. In additon,
the sec_edgar_downloader package that is used to download the 10-K files doesn't always work. This is likely due to the SEC's API either
not giving access or being bombarded by other trying to access those files. I have found that it works best later in the day and at night.
You may have to be patient if you get an error with the HTTP request to download a 10-K file. The error I often get is:

    requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://efts.sec.gov/LATEST/search-index
    
If this happens and you have already downloaded some files, you can comment out line 149 in the extract_10_k.py file, which is:

    dl.get("10-K", cik, after="2015-01-01", download_details=False)
    
and then run the main.py or run.py file again. This way you will not be downloading the files anymore and will still be able
to run predicitions on the files you have already downloaded.

## Visualization

There is a github.io page with a visualization and an explanation of the research pipeline! Please check it out to learn more about
how I got the results. There is a button to the top-right of this text that is under the header **Environments**. Click on the
*github-pages* link and then click on the first *Development* button to pull up the visualization. Scroll down through it to learn about
my paper.