import wget

base_url = "http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-"

for i in range(1, 26):
    url= base_url + str(i) + ".pdf"

    print("donwload:", url)
    try:
        filename = wget.download(url)
        print("\ndownload success:", filename)
    except:
        print("\ndownload failed!")

