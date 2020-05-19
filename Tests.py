import csv

from dictfunctions import LangDistro
from globalVariables import FILE_TEST_PATH, FILE_ANSWER_PATH, LANGS
from DistributionCoreHandler import loadCompleteDistro
from cmpAlgorithms import KullbackLeibner, BhattacharyyaDistance, Hellinger
from generateDistribution import createLangDistribution


#
# Glowny plik testujacy nasze algorytmy
#


def runDistroTests():
    # ladujemy nasz korpus
    dist = loadCompleteDistro()

    sumtests = []
    sumfails = []
    sumtests.append(LangDistro())  # ile testow z danego jezyka bylo
    sumtests.append(LangDistro())
    sumtests.append(LangDistro())

    sumfails.append(LangDistro())  # ile testow nasz algorytm zawalil
    sumfails.append(LangDistro())
    sumfails.append(LangDistro())

    ansline = []

    matrix = []

    for i in range(len(LANGS)):
        matrix.append([])
        for j in range(len(LANGS)):
            matrix[i].append(0)
    print("Runing tests...")
    counter = 0
    # laduemy linijki testow
    with open(FILE_TEST_PATH, "r", encoding='utf-8') as testf:
        testline = testf.readlines()
    # i jezykow w ktorych zostaly napisane - nasze poprawne odpowiedzi
    with open(FILE_ANSWER_PATH, "r", encoding='utf-8') as answerf:
        anslineraw = answerf.readlines()

    # obcinamy zbedny znak konca lini
    for x in anslineraw:
        ansline.append(x.rstrip('\n'))

    # Testujemy!
    for x in range(len(ansline)):

        # Zwiekszamy ilosc testow o 1
        sumtests[0].appenddist(ansline[x])
        sumtests[1].appenddist(ansline[x])
        sumtests[2].appenddist(ansline[x])
        # Przygotowywujemy licznik do testow oblanych z akrutalnego jezyka
        if ansline[x] not in sumfails[0]:
            sumfails[0][ansline[x]] = 0
            sumfails[1][ansline[x]] = 0
            sumfails[2][ansline[x]] = 0

        # analizujemy teskst
        newdistro = createLangDistribution(testline[x])

        # Tutaj wywolujemy nasz algorytm porowujacy rozklad tesktu w nieznanym jezyku z naszymi zapisanymi jezykaim
        # Porownojemy je i zwracamy najbardziej nazwe najbardziej podobnego jezyka
        result1 =BhattacharyyaDistance(newdistro,dist)
        result2 = KullbackLeibner(newdistro, dist)
        result3 = Hellinger(newdistro,dist)

        # Jezeli zly jezyk zwiekszamy liczne porazek
        if result1 != ansline[x]:
            sumfails[0].appenddist(ansline[x])
        if result2 != ansline[x]:
            sumfails[1].appenddist(ansline[x])
        if result3 != ansline[x]:
            sumfails[2].appenddist(ansline[x])

        z1 = LANGS.index(ansline[x])
        z2= LANGS.index(result2)
        matrix[z1][z2] += 1
        counter += 1

        print(f'Tested {counter}/{len(ansline)}')

    avg = []
    for i in range(3):
        mysum = 0
        for x in sumtests[i].keys():
            mysum += (sumtests[i][x] - sumfails[i][x]) / (sumtests[i][x]) * 100
        avg.append(mysum / len(sumtests[i].keys()))

    with open('matrix.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(matrix)

    # Wypisujemy statysyki
    print("Printing Stats...")
    print("Bhattacharyya:")
    for x in sumtests[0].keys():
        print("Lang: \"{0}\" Failed: {1} Tested: {2} Accuracy: {3:0.3f}%".format(x, sumfails[0][x], sumtests[0][x],
                                                                                 ((sumtests[0][x] - sumfails[0][x])
                                                                                  / sumtests[0][x]) * 100))
    print("Avg: " + str(avg[0]) + "%")
    print("KullbackLeibner:")
    for x in sumtests[1].keys():
        print("Lang: \"{0}\" Failed: {1} Tested: {2} Accuracy: {3:0.3f}%".format(x, sumfails[1][x], sumtests[1][x],
                                                                                 ((sumtests[1][x] - sumfails[1][x])
                                                                                  / sumtests[1][x]) * 100))
    print("Avg: " + str(avg[1]) + "%")
    print("Hellinger:")
    for x in sumtests[2].keys():
        print("Lang: \"{0}\" Failed: {1} Tested: {2} Accuracy: {3:0.3f}%".format(x, sumfails[2][x], sumtests[2][x],
                                                                                 ((sumtests[2][x] - sumfails[2][x])
                                                                                  / sumtests[2][x]) * 100))
    print("Avg: " + str(avg[2]) + "%")


if __name__ == "__main__":
    runDistroTests()
