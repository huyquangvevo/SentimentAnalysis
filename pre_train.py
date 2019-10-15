import batchgen

goodfile = "./process/good_tweets.csv"
badfile = "./process/bad_tweets.csv"

x_text, y = batchgen.get_dataset(goodfile, badfile, 5000) #TODO: MAX LENGTH

print(x_text[1])