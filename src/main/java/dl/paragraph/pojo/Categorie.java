package dl.paragraph.pojo;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Categorie {

    private String label;
    private double score;
    public String getScoreArrondi() {
            return String.format("%2.2f", this.score * 100);
        }
}
