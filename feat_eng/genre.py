from csv import DictReader
import omdb,re
from collections import defaultdict
import pickle
omdb.set_default('apikey','68e1c331')
train = list(DictReader(open("../data/spoilers/train.csv", 'r')))

pages = [x['page'] for x in train]
unique_pages = set(pages)

genre_dict = defaultdict()
for page in unique_pages:
    page_title = re.sub(r'([A-Z][a-z]+)', r' \1', page).strip()
    try:
        movie = omdb.title(page_title)
        genre = movie['genre']
        title = movie['title']
        print('##PageTitle:{} ##Title:{} ##Genre:{}'.format(page, title, genre))
        genre_dict[page] = genre
    except:
        print("Exception:", page)
        genre_dict[page] = [""]

pickle.dump(genre_dict, open("genre_dict.p", "wb"))

