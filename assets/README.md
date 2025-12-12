# Short genus names

This is the list of short genus names taken from the ITIS (Integrated Taxonomic Information System) database.

The sqlite database was downloaded from here https://www.itis.gov/downloads/index.html
and then the following query (plus filtering in python) was run to extract the genus names with length equal to 4 characters

```sql
SELECT unit_name1, LENGTH(unit_name1) as name_length
FROM taxonomic_units
WHERE rank_id = 180
ORDER BY name_length, unit_name1
```