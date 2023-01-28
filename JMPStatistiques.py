#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 04 - Un peu de statistiques descriptives pour comprendre les données
#
# Modules necessaires : 
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.1.0
#
# Pour installer un module : 
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------




from collections import Counter
from math import *
import matplotlib.pyplot as plt



class JMPStatistiques:

    def __init__(self,feature):
        self.feature = feature


    def calculMoyenneArithmetique(self):

        n = self.feature.count()
        sommeValeursObservations = 0
        moyenneArithmetique = 0
        for valeurObservation in self.feature:
            sommeValeursObservations = sommeValeursObservations + valeurObservation

        moyenneArithmetique = sommeValeursObservations / n
        return moyenneArithmetique

    def calculMediane(self):
        mediane = 0
        feature = self.feature.sort_values()
        feature = feature.reset_index(drop=True)
        n = self.feature.count()
        pair = False;
        if (n % 2 == 0):
            print("Le nombre d'observations est pair.")
            pair = True

        if pair:
            rang = (n / 2);
            print("RANG = "+str(rang))
            rangPython = rang-1
            valeur1 = feature[rangPython]
            valeur2 = feature[rangPython+1]
            mediane = valeur1 +((valeur2-valeur1)/2)
        else:
            rang = ((n + 1) / 2)
            rangPython = rang - 1
            mediane = feature[rangPython]

        return [mediane, rang]

    def calculMode(self):
        mode = Counter(self.feature)
        return mode

    def calculVarianceEcartType(self):
        n = self.feature.count()
        moyenneArithmetique = self.feature.mean()
        variance = 0
        c3 = 0
        for valeurObservation in self.feature:
            x = valeurObservation
            moy = moyenneArithmetique
            c1 = valeurObservation - moyenneArithmetique
            c2 = c1 * c1
            c3 = c3 + c2

        variance = c3 / (n - 1)

        ecartType = sqrt(variance)

        return ([variance, ecartType])

    def calculDesQuartiles(self,mediane,rangMediane):
        n = self.feature.count()
        sort_feature = self.feature.sort_values()
        sort_feature = sort_feature.reset_index(drop=True)
        q1 = 0
        q2 = mediane
        q3 = 0

        #Calcul Q1
        resteDivision = rangMediane%2
        if (resteDivision != 0):
            q1 = sort_feature[((rangMediane/2)+1)-1]
        else:
            valeurMin = sort_feature[((rangMediane/2)-1)]
            valeurMax = sort_feature[(rangMediane/2)]
            q1 = (valeurMin + ((valeurMax - valeurMin) / 2) + valeurMax) / 2

        # Calcul Q3
        nbdonnees = len(sort_feature)+1
        nbDonneesDepuisMediane = nbdonnees - rangMediane
        resteDivision = nbDonneesDepuisMediane % 2
        if (resteDivision != 0):
            q3 = sort_feature[(rangMediane+ceil(nbDonneesDepuisMediane/2))-1]
        else:
            valeurMinQ3 = sort_feature[(rangMediane+(nbDonneesDepuisMediane/2))-1]
            valeurMaxQ3 = sort_feature[(rangMediane+(nbDonneesDepuisMediane/2))]
            q3 = (valeurMin + ((valeurMax - valeurMin) / 2) + valeurMax) / 2


        return ([q1, q2, q3])


    def critereDeTukey(self, premierQuartile, troisiemeQuartile):

        valeursAberrantesInferieures = []
        valeursAberrantesSuperieures = []
        feature = self.feature.sort_values()
        interquartile = troisiemeQuartile - premierQuartile
        print("Inter-quartile = "+str(interquartile))
        borneInferieure = premierQuartile - (1.5 * interquartile)
        borneSuperieure = troisiemeQuartile + (1.5 * interquartile)

        for valeurObservation in feature:
            if valeurObservation < borneInferieure:
                valeursAberrantesInferieures.append(valeurObservation)

            if valeurObservation > borneSuperieure:
                valeursAberrantesSuperieures.append(valeurObservation)

        valeursAberrantes = valeursAberrantesInferieures + valeursAberrantesSuperieures

        return (valeursAberrantes)



    def visualisation(self,moyenne,mediane,quartile_1,quartile_2,quartile_3):

        plt.subplot(2, 2, 1)
        plt.hist(self.feature)
        plt.title("Histogramme et moyenne")
        plt.axvline(moyenne, color='red', linestyle='dashed', linewidth=1,label = str(moyenne))
        plt.legend(loc='upper right')

        plt.subplot(2, 2, 2)
        plt.hist(self.feature)
        plt.title("Histogramme et médiane")
        plt.axvline(mediane, color='green', linestyle='dashed', linewidth=1,label = str(mediane))
        plt.legend(loc='upper right')

        plt.subplot(2, 2, 3)
        plt.hist(self.feature)
        plt.title("Histogramme et quartiles")
        plt.axvline(quartile_1, color='orange', linestyle='dashed', linewidth=1,label = "Q1: "+str(quartile_1))
        plt.axvline(quartile_2, color='orange', linestyle='dashed', linewidth=1,label = "Q2: "+str(quartile_2))
        plt.axvline(quartile_3, color='orange', linestyle='dashed', linewidth=1,label = "Q3: "+str(quartile_3))
        plt.legend(loc='upper right')

        plt.subplot(2, 2, 4)
        plt.boxplot(self.feature)
        plt.title("Boite à moustaches")
        plt.show()


    def analyseFeature(self):

        print("-----------------------------------------")
        print("      MESURE DE TENDANCE CENTRALE        ")
        print("-----------------------------------------\n")

        print("-- NOMBRE D'OBSERVATIONS --")
        # -Nombre d'observations
        n = self.feature.count()
        print("Nombre d'observations = " + str(n))

        print ("\n-- MIN --")
        valeursTriees = self.feature.sort_values()
        valeursTriees = valeursTriees.reset_index(drop=True)
        print("Valeur minimale : "+str(valeursTriees[0]))

        print ("\n-- MAX --")
        valeursTriees = self.feature.sort_values()
        valeursTriees = valeursTriees.reset_index(drop=True)
        print("Valeur maximale : " + str(valeursTriees[len(valeursTriees)-1]))

        # -Moyenne arithmetique :
        print("\n-- MOYENNE --")
        moyenne = self.calculMoyenneArithmetique()
        print("Moyenne arithmétique  calculée = " + str(moyenne))
        print("> Observations : Si les observations avaient toute la même valeur (répartition équitable) celle-ci serait de " + str(moyenne))

        # -Moyenne arithmetique :
        print("\n-- MEDIANE --")
        mediane = self.calculMediane()
        print("Mediane  calculée = " + str(mediane[0]))
        print("> Observations : La valeur se trouvant au milieu des observations est de :" + str(mediane[0]))
        print("La répartition est de : " + str(mediane[1]) + " valeurs de chaque côté de la médiane")

        # -Mode
        print("\n-- MODE --")
        mode = self.calculMode()
        print(mode)
        print("> Observations : le mode permet de déterminer les valeurs les plus souvent observées")


        print("\n\n-----------------------------------------")
        print("      MESURE DE DISPERSION        ")
        print("-----------------------------------------\n")
        print("-- ETENDUE --")
        print ("Etendue de la série = "+str(valeursTriees[len(valeursTriees)-1]-valeursTriees[0]))
        varianceEcartType = self.calculVarianceEcartType()

        print("\n-- VARIANCE --")
        print("Variance calculée = " + str(varianceEcartType[0]))

        print("\n-- ECART TYPE --")
        print("Ecart type calculé = " + str(varianceEcartType[1]))
        ecartType = varianceEcartType[1]
        print("68 % des valeurs des observations se situent entre " + str(moyenne - ecartType) + " et " + str(
            moyenne + ecartType))
        print("95 % des valeurs des observations se situent entre " + str(moyenne - (ecartType * 2)) + " et " + str(
            moyenne + (ecartType * 2)))
        print("99 % des valeurs des observations se situent entre " + str(moyenne - (ecartType * 3)) + " et " + str(
            moyenne + (ecartType * 3)))

        print("\n\n-----------------------------------------")
        print("      QUARTILES        ")
        print("-----------------------------------------\n")
        quartiles = self.calculDesQuartiles(mediane[0],mediane[1])
        print("25% des observations ont une valeur inférieure à " + str(quartiles[0]))
        print("50% des observations ont une valeur inférieure à " + str(quartiles[1]))
        print("75% des observations ont une valeur inférieure à " + str(quartiles[2]))


        print("\n\n-----------------------------------------")
        print("      DETECTION VALEURS ABERRANTES        ")
        print("-----------------------------------------\n")
        print("> Critère de Tukey")
        valeursAberrantes = self.critereDeTukey(quartiles[0], quartiles[2])
        print(" Nombre de valeurs aberrantes : " + str(len(valeursAberrantes)))
        print(" Valeurs :" + str(valeursAberrantes))


        print("\n\n-----------------------------------------")
        print("      VISUALISATION        ")
        print("-----------------------------------------\n")
        print("Generation des graphiques...")
        self.visualisation(moyenne,mediane[0],quartiles[0],quartiles[1],quartiles[2])