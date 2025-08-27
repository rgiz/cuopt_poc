# Data requirements

- Raw RSL data to be loaded from UI frontend
- Raw RSL data to be cleaned, adding postcodes, lat/long and fill in any blanks
- Raw RSL data to add a field for priority, to be determined by the "Load Type" and categorized in priority levels 1 to 5
- Distance Miles Matrix, Location Index, Driver States and Time Minutes Matrix to be generated from the cleaned RSL
- RSL data will always have mileages and minutes data for any travel leg, but postcodes and lat/long are not included in the raw dataset

# UI Config requirements - Settings UI

- Penalties and costs to be configured in panel (default in dot env)
- SLA paramaters to be configued in panel - delay penalties for each priority level
- Max number of cascades (cascading driver reassignments per solution)
- Driver's hours constraints (hard constraints)

# UI Data Requirements

- Facility to upload new RSL and generate matrices
- Matrices to be sent to backend for storage and overwritten when new RSL uploaded

# UI Disruption Manager (Main Interface)

- Pick start and end location for additional load, select either "depart after" or "arrive before" parameter and specify datetime
- Solve button sends query to cuOpt
- cuOpt processes query
- UI shows list of possible solutions (collapsed) with weighted cost impact and full implementation details. This will entail printing the old/new full schedule for any affected driver.
