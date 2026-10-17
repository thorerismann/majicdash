#!/usr/bin/env python3
"""Build the versioned corpus tables directly from original WordPress HTML.

Run with the majic2 environment. Default outputs live in assets/corpus; pass
--legacy-exports to refresh the older assets/corpus_matrix.* locations as well.
The application is not imported or modified. See assets/corpus/README.md.
"""

from corpus_tools.pipeline import main

if __name__ == "__main__":
    main()
