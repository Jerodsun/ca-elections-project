# QGIS Buisness Logic

There are 8,507 Census Tracts and 23,194 Block Groups in California.

### Census Bureau Shapefiles with Precinct Shapefiles 
#### Tract-Level Analysis

- Import census shapefiles: census tract coordinate matching system EPSG:4269 - NAD83 to EPSG:4326 - WGS84 with 2m accuracy
- Import individual Shapefiles for 2016 precinct maps: EPSG:4326 - WGS 84
- Merge Vector Layers for precinct maps, save.
- Calculate new field `$area` for both: `area_c` (census area) and `area_p` (precinct area)
- `UNION` two layers
- Note error count
- Calculate new field `$area` for each union area: `area_v`
- Export 


Name is not distinct. `select distinct NAMELSAD, COUNTYFP from census_tract_area;`

### ArcGIS Shapefiles with Precinct Shapefiles
#### Block-Level Analysis

- Import census shapefile: `census_block_ca` EPSG:4326 - WGS84
- Import individual Shapefiles for 2016 precinct maps: EPSG:4326 - WGS 84
- Merge Vector Layers for precinct maps
- Calculate new field `$area` for both: `area_c` (census area) and `area_p` (precinct area)
- `UNION` two layers
- Note error count
- Calculate new field `$area` for each union area: `area_v`
- Export 

- take all shape

### Punt to Python process:

Calculate new field: `pct_precinct =  v_area / p_area`

Percent of the precinct - we will use that to mask over the precinct results.



Business logic:

Ok at this point I don't even know how the order ended up like this but I think this is what it is:

A: QGIS this was the first one created - tracts

Q: New business logic needed - block group Check the business logic here

S: New one direct from statewidedatabase - ignore


### Ok restart: to merge 2016 and 2018 data, first merge the precincts in QGIS. Some are not a direct match for some reason...

### Ok conclusion is that it;s bad very very bad

Visualize 2016- 2018 shift: MEDSL clean data

### So abandon the 2016 - 2018 collection based on Census data... someone must collect it soon.



San Diego only:

Downloaded from :

Process:


After UNION:

Delete these: 
NOT ("area_combined" > 1000 and "area_2016" > 0 and "area_2018" > 0)
nulls remain