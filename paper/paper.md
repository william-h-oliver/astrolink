---
title: 'AstroLink: A Python package for robust and interpretable astrophysical clustering'
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

`AstroLink` is a general purpose clustering algorithm built to extract meaningful hierarchical structure from astrophysical data sets. In practice `AstroLink` rarely requires any parameter tuning before application, nevertheless, it has a small number of intuitive-to-adjust parameters should this be necessary. As such, it is readily capable of finding an arbitrary number of arbitrarily-shaped clusters (and their structural relationship within the broader hierarchy) from point-based data sets with arbitrary sizes and dimensionalities. Clusters found by `AstroLink` are defined as being statistically distinct overdensities when compared to both their surrounds and to the noisy density fluctuations within the data set.

# Statement of need

Extracting astrophysically-relevant structure from both simulations and observations is a necessary hurdle towards understanding the physics of our Universe. With progressively larger data sets on the horizon and a continued deeper understanding of the structure they accommodate, astrophysical clustering algorithms have seen lasting attention and growth. Functionally however, many of these algorithms are similar and will often fall into one of a few common algorithm types. Simulation-specific structure finders &mdash; e.g. `SubFind` [@SubFind], `AHF` [@AHF], `ROCKSTAR` [@ROCKSTAR], `VELOCIraptor` [@VELOCIraptor] &mdash; are typically density-based and tend to be built upon the somewhat inflexible `Spherical Overdensity` [@SO] and/or `Friends-Of-Friends` [@FOF] algorithms. As such these algorithms adopt additional techniques to ensure the robustness of the resultant clusters. Observation-specific structure finders &mdash; e.g. `Matched Filter` [@MatchedFilter], the `xGC3` suite [@GC3; @mGC3; @xGC3], `StreamFinder` [@StreamFinder] &mdash; are typically model-based and require explicit physical knowledge/assumptions for their operation but are restricted to finding specific structure types as a consequence. While many of these specific-use codes could also be applied to both simulated and observational data sets pending minor changes to their code-base, there are some generalised astrophysical structure finders that may be applied in both contexts without the necessity to make code-base changes &mdash; e.g. `EnLink` [@EnLink], `FOPTICS` [@FOPTICS], `Halo-OPTICS` [@HaloOPTICS], `CluSTAR-ND` [@CluSTARND]. These codes are generally density-based like simulation-specific finders (which suits the nature of the hierarchical structure formation inherent within our Universe) but do not rely on additional physical knowledge/assumptions about the data (which may or may not be possible to make in every given application context). However, many of these codes are neither open-source nor freely available (as is also true for many of the specific-use codes) and so potential users have often resorted to non-astrophysical clustering algorithms which are typically ill-suited for the task.

By comparison, `AstroLink` is an open-source and generalised astrophysical structure finder that does not suffer from the same drawbacks as many of the existing codes in the field. As the algorithmic successor to `CluSTAR-ND` and `Halo-OPTICS`, `AstroLink` is efficient and versatile, astrophysically and statistically robust, and is capable of extracting a wide range of structure types from any size and dimensionality data set [@AstroLink]. In addition to being a state-of-the-art clustering algorithm, the `AstroLink` output can also be used to generate a visualisation of the clustering structure as shown in \autoref{fig:astrolink_example}. With this, the user can explore the structure within the input data in a way that can not be done with the output of other astrophysical clustering algorithms.

![An example of the `AstroLink` clustering output on a 2D toy data set. The top panel shows the data coloured by which cluster the points are found to be a part of. The bottom panel shows the corresponding ordered-density plot which can be used to visualise the clustering structure of the data.\label{fig:astrolink_example}](astrolink_example.png){ width=80% }

# Acknowledgements

WHO and TB acknowledge financial support from the Carl-Zeiss-Stiftung. The authors would also like to thank the Scientific Software Center at the Interdisciplinary Center for Scientific Computing of Heidelberg University for its support and guidance during the development of this package.

# References