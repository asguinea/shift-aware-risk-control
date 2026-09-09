# Dataset provenance and usage

No external dataset or pretrained model weight is committed to this repository.
Users are responsible for obtaining each resource from its authoritative source
and complying with its license, access conditions, and intended-use restrictions.

## Supported sources

| Dataset | Role in the toolkit | Authoritative source |
| --- | --- | --- |
| CIFAR-10-C | Controlled image corruptions | [Zenodo record 2535967](https://zenodo.org/records/2535967) |
| CIFAR-100-C | Controlled image corruptions | [Zenodo record 3555552](https://zenodo.org/records/3555552) |
| Camelyon17-WILDS | Natural medical-image shift | [WILDS datasets](https://wilds.stanford.edu/datasets/) |
| CivilComments-WILDS | Natural text and subgroup shift | [WILDS datasets](https://wilds.stanford.edu/datasets/) |
| FMoW-WILDS | Natural satellite-image shift | [WILDS datasets](https://wilds.stanford.edu/datasets/) |
| Office-Home | Cross-domain image shift | [Office-Home dataset](https://hemanthdv.github.io/officehome-dataset/) |

## Storage and downloads

- `data/` is ignored by Git and is the default local data root.
- `outputs/caches/` is ignored and stores derived prediction caches.
- CIFAR-C and WILDS download paths require an explicit command-line download flag.
- Office-Home uses a manual directory layout with `Art`, `Clipart`, `Product`, and
  `Real_World` subdirectories.
- Pretrained torchvision weights may trigger a separate download on first use.

Do not commit raw data, derived examples, user-provided text, access tokens, or
model weights. When sharing a result, retain provenance identifiers, source URLs,
checksums when available, configuration files, and the generated run manifest.

## Sensitive data

Some supported datasets contain medical images or text about demographic groups.
This toolkit does not remove the obligations attached to those sources. Review
the source documentation, restrict access where required, and avoid exporting
row-level content unless redistribution is explicitly permitted.
