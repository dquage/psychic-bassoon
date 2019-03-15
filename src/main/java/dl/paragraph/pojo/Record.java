package dl.paragraph.pojo;

import com.sun.istack.internal.NotNull;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Record {

    public enum Type {
        RECETTE, DEPENSE;
        public static Type fromString(String str) {
            return str.equals("R") ? RECETTE : DEPENSE;
        }

        @Override
        public String toString() {
            return this == RECETTE ? "R" : "D";
        }
    }

    private Type type = Type.DEPENSE;
    private Apollon apollon;
    private String categorie;
    private String numeroCompte;

    @NotNull
    private String libelle;
    @NotNull
    private double montant;

    public boolean isRecette() {
        return type == Type.RECETTE;
    }

    public boolean isDepense() {
        return !isRecette();
    }
}
