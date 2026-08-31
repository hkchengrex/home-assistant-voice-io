"""Keep the checked-in documentation independent of its build date."""

import gzip
from pathlib import Path


def on_post_build(config, **kwargs):
    # MkDocs adds today's date to the gzip header even without sitemap lastmod.
    sitemap = Path(config.site_dir) / "sitemap.xml"
    with sitemap.with_suffix(".xml.gz").open("wb") as output:
        with gzip.GzipFile(
            filename="sitemap.xml", mode="wb", fileobj=output, mtime=0
        ) as compressed:
            compressed.write(sitemap.read_bytes())
