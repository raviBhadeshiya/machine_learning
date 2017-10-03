from csv import DictReader
from collections import defaultdict
import pickle
import omdb,re
omdb.set_default('apikey','68e1c331') # Courtesy of peer

train = list(DictReader(open("../data/spoilers/train.csv", 'r')))

pages = [x['page'] for x in train]

unique_pages = set(pages)

genre_dict = defaultdict()
for page in unique_pages:
    page_title = re.sub(r'([A-Z][a-z]+)', r' \1', page).strip()
    try:
        movie = omdb.title(page_title)
        genre = movie['genre']
        genre_dict[page] = genre
    except:
        print("Exception:", page)
        genre_dict[page] = [""]

    print('##PageTitle:{} ##Genre:{}:'.format(page, genre_dict[page]))

pickle.dump(genre_dict, open("genre_dict.p", "wb"))

genre_dict = pickle.load(open("genre_dict.p", "rb"))

for keys,values in genre_dict.items():
    print(values)