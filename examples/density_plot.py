"""
Simple example to visualize a density plot
"""
from summaries.analysis import DensityPlot


if __name__ == '__main__':

    example_reference = """
Die erste Herrenmannschaft des Rugby Club Aachen ist im Jahr 2012 erstmals in die 1. Bundesliga aufgestiegen, spielt jedoch mittlerweile erfolgreich in der 2. Bundesliga.
Im Hockey spielt die erste Herrenmannschaft des Aachener Hockey-Clubs seit 2019 wieder in der Regionalliga.

Seit 2005 besteht der American-Football-Club Aachen Vampires, der seine Heimspiele im Ludwig-Kuhnen-Stadion in Burtscheid austrägt und derzeit (2018) in der Oberliga Nordrhein-Westfalen antritt.

Aufwärts Aachen spielt in der ersten Schachbundesliga.
Der Aachener SV von 1856, der älteste Schachverein im Aachener Schachverband, ist mit seiner ersten Mannschaft in der 2. Bundesliga-Gruppe West vertreten.
In Aachen sind noch sechs weitere Schachvereine ansässig.

Mit der Karlsschützengilde vor 1198 Aachen e. V. besitzt Aachen den ältesten Verein Deutschlands, der ursprünglich für den Schutz der Aachener Pfalzkapelle zuständig war, sich aber mittlerweile erfolgreich auf dem Gebiet der olympischen Disziplinen im Sportschießen spezialisiert hat und der dazu über einen anerkannten Leistungsstützpunkt in Eilendorf verfügt.

Weitere national und international erfolgreiche Aachener Sportvereine sind die Aachener Schwimmvereinigung 06 im Schwimmen, der SV Neptun Aachen 1910 im Kunst- und Turmspringen, der Burtscheider Turnverein im Trampolinturnen, der BTB Aachen im Handball und der Allgemeine Turnverein Aachen im Rhönradturnen.
"""
    extracted_sentence = "Aufwärts Aachen spielt in der ersten Schachbundesliga."

    plot = DensityPlot()
    plot.plot([example_reference], [extracted_sentence])
