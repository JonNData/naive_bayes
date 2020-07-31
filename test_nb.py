# Run `pytest` in the terminal
from naive_bayes import MNNaiveBayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# Data will be from reddit. 
# Train on 10 r/worldnews titles and 10 r/aww titles
# test on 2 r/aww and 2 r/worldnews
# category_0 = r/worldnews = 0 
# category_1 = r/aww = 1
worldnews = ["Uighur group calls for China to lose 2022 Games over 'genocide'", 
     "Polish Towns That Declared Themselves ‘L.G.B.T. Free’ Are Denied E.U. Funds",
     "Michelle Bolsonaro, Brazil's First Lady, Tests Positive For Coronavirus",
     "Border officials crack down on Americans travelling through B.C. to Alaska",
     "Hong Kong bans 11 pro-democracy figures from legislative election | Hong Kong Free Press HKFP",
     "The 3 women who have brought COVID into Queensland have been charged with falsifying documents and fraud",
     "UK KFC admits a third of its chickens suffer painful inflammation - Fast food giant KFC has laid bare the realities of chicken production after admitting to poor welfare conditions among its suppliers.",
     "Chile picks Japan's trans-Pacific cable route in snub to China",
     "Hackers post fake stories on real news sites 'to discredit Nato'",
     "Prostate cancer can be detected by a new blood test which also reveals the severity of the disease with 99 per cent accuracy"
    ]
aww = [
       "This little cutie climbed up on me while I applied to adopt her",
       "Here is a happy duckling to make your day better!",
       "Adorable cutie",
       "12 years ago she came running up to me on a dirt road and sat on my foot clinging to my ankle crying. Today I present to you my kitty Izzy.",
       "Very talented so CUTE Otter",
       "A dog at the shelter I work at is teaching me how to smile.",
       "The best seat in the house",
       "A Stork couple celebrating their first egg ",
       "She turned 6 last week. Everyone still thinks she's a kitten.",
       "My gf and I rescued this little guy today.... meet max everyone"
]
X = worldnews + aww
y = [0]*10 + [1]*10

X_test = [
          "Toronto emerging as tech superpower as immigrants choose Canada over US",
          """Egypt imprisons female TikTok influencers: A court in Cairo has sentenced six young female bloggers to prison for up to two years — not for political offenses, but for violating "public morals." Activists have called the ruling an "outrageous attack on civil liberties.""",
          "The mixed kitten seeds grew well this year.",
          "My wife just sent me this photo of our cat at the vet. Safe to say she’s a little scared."
]

def test_nb_class():
    # fit an instance and test it against 2 docs of each
    nb = MNNaiveBayes()
    nb.fit(X, y)
    assert nb.predict(X[0:2] + X[-3:-1]) == [0, 0, 1, 1]


def test_sklearn():
    # Need to preprocess manually 
    cv = CountVectorizer(strip_accents='ascii',
                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',
                     lowercase=True, stop_words='english')
    X_train_cv = cv.fit_transform(X)
    X_test_cv = cv.transform(X_test)
    # sklearn naive bayes
    MNB = MultinomialNB()
    MNB.fit(X_train_cv, y)
    assert (MNB.predict(X_test_cv) == [0, 1, 1, 1]).all()

    # our naive bayes from scratch
    nb = MNNaiveBayes()
    nb.fit(X, y)
    assert nb.predict(X_test) == [0, 1, 1, 1]

