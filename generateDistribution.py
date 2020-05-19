import argparse
from dictfunctions import LangDistro
from globalVariables import NO_ALFACHARACTERS, MINIMAL_OCCURANCE

#
# W tym skrypcie tworzymy  rozklady z pliku lub lini
#


# Czy nasz znak to literka.
def standardchar(c):
    if c not in NO_ALFACHARACTERS:
        return True
    else:
        return False

#Tworzymy rozklad z pojedynczej linijki tekstu
def createLangDistribution(line, distribution=None):
    if distribution is None:
        flag =True
        distribution = LangDistro()
    else:
        flag = False
    for c in line:
        if standardchar(c):
            distribution.appenddist(c.lower())
        else:
            pass

    if flag:
        distribution.sort()
        distribution.makePercentage()

    return distribution

#tworzymy rozklad z calego pliku
def createDistroFormFile(startpath):
    distribution = LangDistro()

    with open(startpath, encoding='utf-8') as f:
        for line in f:
            createLangDistribution(line, distribution)

    distribution.removeBelowMinimal(MINIMAL_OCCURANCE)
    distribution.sort()
    distribution.makePercentage()

    return distribution
    # Debug info
    #
    # for i in distribution:
    #    print("Keys: {0} Count:{1} Percentage {2:0.4f}%".format(i, distribution[i], avgdict[i]))
    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="file path containing data lang",
                        type=str)
    args = parser.parse_args()
    createDistroFormFile(args.fname)
