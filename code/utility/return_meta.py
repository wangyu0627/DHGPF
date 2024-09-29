
def return_meta(dataset):
    if dataset == "Movielens":
        meta_paths = {
            "user": [["um", "mu"]],
            # "user": [["ua", "au"]],
            # , ["ua", "au"], ["uo", "ou"], ["mu", "ua", "au", "um"]
            # "user": [["um", "mu"],["um", "mg", "gm", "mu"]],
            "movie": [["mu", "um"], ["mg", "gm"]],
        }
        user_key = "user"
        item_key = "movie"
    elif dataset == "Amazon":
        meta_paths = {
            "user": [["ui", "iu"]],
            # , ["ui", "ib", "bi", "iu"], ["iv", "vi"], ["ui", "ic", "ci", "iu"]
            "item": [["iu", "ui"], ["ic", "ci"], ["ib", "bi"]],
        }
        user_key = "user"
        item_key = "item"
    elif dataset == "Yelp":
        meta_paths = {
            "user": [["ub", "bu"]],
            # , ["ub", "bu", "ub", "bu"]
            # "user": [["uc", "cu"], ["ub", "bu", "ub", "bu"]],
            "business": [["bu", "ub"], ["bc", "cb"], ["bi", "ib"]],
        }
        user_key = "user"
        item_key = "business"
    elif dataset == "dblp":
        meta_paths = {
            # "paper": [["pa", "ap"], ["pc", "cp"], ["pt", "tp"]],
            "paper": [["pa", "ap"], ["pc", "cp"]],
            # "author": [["ap", "pa"], ["al", "la"]],
            "author": [["ap", "pa"], ["al", "la"]],
        }
        user_key = "paper"
        item_key = "author"
    elif dataset == "Douban Book":
        meta_paths = {
            "user": [["ub", "bu"], ["ul", "lu"]],
            "book": [["bu", "ub"], ["ba", "ab"]],
        }
        # , ["ug", "gu"], ["ul", "lu"]
        # , ["ba", "ab"], ["by", "yb"]
        user_key = "user"
        item_key = "book"
    else:
        print("Available datasets: Movielens, amazon, yelp.")
        raise NotImplementedError
    return meta_paths,user_key,item_key