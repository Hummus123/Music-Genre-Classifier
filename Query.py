import requests as r
import flickr_api
from PIL import Image
import numpy as np
genres = ["Rock", "Pop", "Classical", "Hip Hop", "Rythm and blues", "Country", "Jazz", "Electronic"]
print(f"Using {len(genres)}")
genres = [i + " music" for i in genres]
perpage = 10
total_requested = 150
needed = int(total_requested / perpage)
flickr_api.set_keys(api_key = 'a5956ebe13db2952d90bb91962331db3', api_secret = 'cb791e6b7dea5d51')

#saves total_requested images for each genre. 
def getimg(query: str, perpage: int, needed:int, save = False):
    for i in range(needed):
        search = flickr_api.Photo.search(text = query, per_page = perpage, page = i+1, content_types = 4, sort = 'relevance')
        print(search)
        for b in search:
            print(b)
            if save:
                b.save(query + str(i) + b.id)

#for i in genres:
    #getimg(i, perpage, needed, save = True)