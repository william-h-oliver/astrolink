---
title: 'AstroLink: A Python package for robust astrophysical clustering'
tags:
  - Python
  - Numba
  - Astronomy
  - Astrophysics
  - Clustering
  - Unsupervised learning
  - Machine learning
authors:
  - name: William H. Oliver
    orcid: 0009-0008-1180-537X
    affiliation: "1, 2"
  - name: Pascal J. Elahi
    orcid: 0000-0002-6154-7224
    affiliation: 3
  - name: Geraint F. Lewis
    orcid: 0000-0003-3081-9319
    affiliation: 4
  - name: Tobias Buck
    orcid: 0000-0003-2027-399X
    affiliation: "1, 2"
affiliations:
 - name: Universit&auml;t Heidelberg, Interdisziplin&auml;res Zentrum f&uuml;r Wissenschaftliches Rechnen, Im Neuenheimer Feld 205, D-69120 Heidelberg, Germany
   index: 1
 - name: Universit&auml;t Heidelberg, Zentrum f&uuml;r Astronomie, Institut f&uuml;r Theoretische Astrophysik, Albert-&Uuml;berle-Stra&szlig;e 2, D-69120 Heidelberg, Germany
   index: 2
 - name: Pawsey Supercomputing Research Centre, 1 Bryce Avenue, Kensington, WA 6151, Australia
   index: 3
 - name: Sydney Institute for Astronomy, School of Physics A28, The University of Sydney, NSW 2006, Australia
   index: 4
date: 21 July 2024
bibliography: paper.bib
---

# Summary

`AstroLink` is a general purpose clustering algorithm built to extract meaningful hierarchical structure from astrophysical data sets. In practice `AstroLink` rarely requires any parameter tuning before application, nevertheless, it has a small number intuitive-to-adjust parameters should this be necessary. As such, it is readily capable of finding an arbitary number of arbitrarily shaped clusters (and their structural relationship within the broader hierarchy) from arbitrarily defined point-based data sets. Clusters found by `AstroLink` are defined as being statistically distinct overdensities when compared to their surrounds and to the noisy density fluctuations within the data set.

# Statement of need

Extracting astrophysically-relevant structure from both simulations and observations is a necessary hurdle towards understanding the physics of our Universe. With progressively larger data sets on the horizon and a continued deeper understanding of the structure they accommodate, astrophysical clustering algorithms have seen lasting attention and growth. Functionally however, many of these algorithms are similar and will typically fall into one of a few common algorithm types. Simulation-specific structure finders &mdash; e.g. `SubFind` [@SubFind], `AHF` [@AHF], `ROCKSTAR` [@ROCKSTAR] &mdash; are generally density-based and tend to be based on the somewhat inflexible `Spherical Overdensity` [@SO] and/or `Friends-Of-Friends` [@FOF] algorithms. As such these algorithms require additional techniques to ensure the robustness of the resultant clusters. Observation-specific structure finders &mdash; e.g. `Matched Filter` [@MatchedFilter], the `xGC3` suite [@GC3; @mGC3; @xGC3], `StreamFinder` [@StreamFinder] &mdash; are generally model-based and require explicit physical knowledge for their operation but are restricted to specific structure types as a consequence. While there are some more generalised astrophysical structure finders, that may be applied to both simulations and observations, have been explored in the literature &mdash; e.g. `EnLink` [@EnLink], `FOPTICS` [@FOPTICS], `Halo-OPTICS` [@HaloOPTICS], `CluSTAR-ND` [@CluSTARND] &mdash; these codes are neither open-source nor freely available (as is also true for many of the specific-use codes).

By comparison, `AstroLink` is an open-source and generalised astrophysical structure finder that does not suffer from the same drawbacks as many of the existing codes in the field. `AstroLink` is efficient and versatile, astrophysically and statistically robust, and is capable of extracting a wide range of structure types from any size and dimensionality data set [@AstroLink]. In addition to being a state-of-the-art clustering algorithm, the `AstroLink` output can also be used to generate a visualisation of the clustering structure as shown in \autoref{fig:astrolink_example}. With this, the user can explore the structure within the input data in a way that can not be done with the output of other astrophysical clustering algorithms.

![An example of the `AstroLink` clustering output on a 2D toy data set. The top panel shows the data coloured by which cluster the points are found to be a part of. The bottom panel shows the corresponding ordered-density plot.\label{fig:astrolink_example}](astrolink_example.png){ width=80% }

# Acknowledgements

WHO and TB acknowledge financial support from the Carl-Zeiss-Stiftung. The authors would also like to thank the Scientific Software Center at the Interdisciplinary Center for Scientific Computing of Heidelberg University for its support and guidance during the development of this package.

# References