package dl.paragraph.pojo;

import lombok.Builder;

@Builder
public class LabelCount {
    public Apollon apollon;
    public int count;

    @Override
    public boolean equals(Object obj) {
        return this.apollon == ((LabelCount) obj).apollon;
    }

    @Override
    public String toString() {
        return "[" + apollon + " - " + count + "]";
    }
}
