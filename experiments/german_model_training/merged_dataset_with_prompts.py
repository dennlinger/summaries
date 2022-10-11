"""
This introduces a torch-compatible dataset class that combines the samples from all the available German models.
Currently, we consider the following:
- MLSUM (de)
- MassiveSumm (de)
- WikiLingua (de)
- Klexikon
- Swisstext/GeWiki
- LegalSum
- EUR-Lex-Sum (Aumiller et al.'s version)

For the respective datasets, we also add prompt to differentiate between the different styles to encourage the model
to learn different summarization patterns across data. We use the following patterns:

- Prompt: "Zusammenfassung News:"; Used sources: MLSUM + Mixin MassiveSumm
    Due to the data quality (and bigger size) of MassiveSumm, we use aggressive filtering and match
     the number of available MLSUM samples with samples from MassiveSumm.
- Prompt: "Zusammenfassung Instruktionen:"; Used sources: WikiLingua
- Prompt: "Zusammenfassung Wikipedia:"; Used sources: Swisstext/GeWiki + Mixin of Klexikon.
    Due to the size of Klexikon, we use all available samples.
- Prompt: "Vereinfachte Zusammenfassung Wikipedia:"; Used sources: Klexikon
- Prompt: "Zusammenfassung Gerichtsentscheid:"; Used sources: LegalSum
- Prompt: "Zusammenfassung Legislatur:"; Used sources: EUR-Lex-Sum

We further use additional samples from each dataset to bake in respective behavior when querying without prompts.
We randomly sample max(2000, 0.1 * num_samples) samples per dataset to balance the exposure to different domains.
"""

from torch.utils.data.dataset import Dataset


class GermanSummarizationDataset(Dataset):
    pass