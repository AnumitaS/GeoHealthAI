## This project is an end-to-end, spatial AI decision-support platform designed to analyze population growth and optimize healthcare facility placement across West Bengal. By combining spatial analysis, demographic modeling, and automated policy reporting, the system identifies critical healthcare coverage gaps and recommends optimal locations for new facilities.

### Core Features & System Architecture

####    Data Parsing & Harmonization: Extracts raw healthcare facility records directly from state GeoJSON datasets, standardizes field schemas, and maps spatial coordinates to standard tabular formats.

####    Smart Geocoding Engine: Utilizes automated browser scraping and fallback strategies to resolve missing addresses, validate boundary limits, and eliminate coordinate inaccuracies.

####    Geospatial & Transport Analysis: Integrates high-resolution census village data, national highway networks, railway tracks, and administrative boundaries to compute accessibility metrics.

####    Demographic Projection Engine: Models population changes between 2011 and 2026 at the village level to detect areas experiencing severe population expansion against existing facility capacity.

####    AI-Driven Site Placement Optimization: Calculates dynamic spatial coverage gaps and deploys optimization algorithms to suggest exact coordinates for new health facilities.

####    Automated Deliverables & Policy Briefs: Generates comprehensive system documentation, high-resolution transport and demographic maps, structured CSV recommendations, and policy-ready PDF executive briefs.
