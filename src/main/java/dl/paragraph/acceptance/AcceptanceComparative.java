package dl.paragraph.acceptance;

import dl.paragraph.pojo.Resultat;

/**
 * Accepte selon la précision de tous les éléments.
 * Soit la précision du meilleur résultat dépasse une limite acceptable.
 * Soit la précision du meilleur résultat dépasse un minimum acceptable mais les autres résultats ne dépasse pas un palier.
 */
public class AcceptanceComparative implements Acceptance {

    private int precisionAccepte = 80;
    private int precisionMin1 = 60;
    private int palier1 = 40;
    private int precisionMin2 = 40;
    private int palier2 = 20;

    public AcceptanceComparative() {
    }

    public AcceptanceComparative(int precisionAccepte, int precisionMin1, int palier1, int precisionMin2, int palier2) {
        this.precisionAccepte = precisionAccepte;
        this.precisionMin1 = precisionMin1;
        this.palier1 = palier1;
        this.precisionMin2 = precisionMin2;
        this.palier2 = palier2;
    }

    @Override
    public String accept(Resultat resultat) {

        double score1 = resultat.getCategorie1().getScore() * 100;
        double score2 = resultat.getCategorie2().getScore() * 100;
        double score3 = resultat.getCategorie3().getScore() * 100;

        boolean accepte = (score1 > precisionAccepte)
                || (score1 > precisionMin1 && score2 < palier1 && score3 < palier1)
                || (score1 > precisionMin2 && score2 < palier2 && score3 < palier2);

        return accepte ? resultat.getCategorie1().getLabel() : null;
    }
}
