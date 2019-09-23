from html.parser import HTMLParser
import re
import wget

# Parse a link
class MyHTMLParser(HTMLParser):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for (key, value) in attrs:
                if key == "href":
                    if re.match(self.pattern, value):
                        self.links.append(value)
                        # print(value)

    def handle_endtag(self, tag):
        pass

# Download from a url
def dowload(url):
    try:
        file_name = wget.download(url)
        print("ok:", file_name, "from:", url)
        return file_name
    except:
        print("error:", url)
        return None

# Read from a file
def read_as_text(file_name):
    file = open(file_name)
    try:
        text = file.read()
        return text
    except:
        print("error:", file_name)
        return None
    finally:
        file.close()

if __name__ == "__main__":
    base_url = "http://cs231n.stanford.edu/slides/2019/"

    file_name = dowload(base_url)
    text = read_as_text(file_name)

    pattern = "[a-zA-Z0-9\-\_]*\.pdf"
    parser = MyHTMLParser(pattern)
    parser.feed(text)

    for link in parser.links:
        full_path_link = base_url + link
        print(full_path_link)
        dowload(full_path_link)
