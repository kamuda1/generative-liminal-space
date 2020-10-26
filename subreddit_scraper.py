import praw
import requests
import os
from config import client_id, client_secret

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent='useragent',
                     )

def save_image_from_url(image_url, save_folder='', image_title=None):

    if 'gallery' in image_url:
        return

    if 'jpg' not in image_url and 'png' not in image_url:
        return

    if image_title is None:
        image_title = image_url.split('.')[-2]
        image_title = image_title.split('/')[-1]
    img_data = requests.get(image_url).content

    image_title = os.path.join(save_folder, image_title)
    with open(image_title + '.png', 'wb') as handler:
        handler.write(img_data)


submission = reddit.subreddit("LiminalSpace").hot(limit=1000)
for item_index, item in enumerate(submission):
    saved_file_name = 'image_num_' + str(item_index)
    save_image_from_url(item.comments._submission.url, 'scraped_data', saved_file_name)
