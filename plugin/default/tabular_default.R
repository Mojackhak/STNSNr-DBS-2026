
# helper: build "('a', 'b')" pair levels in the same order as
# [(a,b) for a in x for b in x]
make_pair_levels_tuple <- function(x) {
  x <- as.character(x)
  g <- expand.grid(b = x, a = x, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  sprintf("('%s', '%s')", g$a, g$b)
}

make_pair_levels_string <- function(x) {
  x <- as.character(x)
  g <- expand.grid(b = x, a = x, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  sprintf("%s-%s", g$a, g$b)
}

region <- c("SNr", "STN")
regions_all <- c(region, make_pair_levels_string(region),
                 make_pair_levels_tuple(region))

side <- c("Left", "Right", "Bilat")
sides_all <- c(side, make_pair_levels_string(side),
               make_pair_levels_tuple(side))

hemi <- c("LAH", "MAH", "Bilat")
hemis_all <- c(hemi, make_pair_levels_string(hemi), 
               make_pair_levels_tuple(hemi))

lat <- c("Ipsi", "Contra", "Bilat")
lats_all <- c(lat, make_pair_levels_string(lat), 
              make_pair_levels_tuple(lat))

df_category <- list(
  Region     = regions_all,
  Side       = sides_all,
  Side_pose  = sides_all,
  Side_cycle = sides_all,
  Side_stim  = sides_all,
  Hemi       = hemis_all,
  Lat        = lats_all,
  Loc_lat    = lats_all,
  Pose       = c("Sit", "Stand", "Walk", "Turn", "Gait", "FoG"),
  Med        = c("Off", "On"),
  Task       = c("Single", "Dual"),
  Polarity   = c("Anodal", "Cathodal", "Bipolar"),
  Phase      = c("BL", "Pre", "ONS", "SUS", "OFS", "PST", "HS", "TO", "Stance", "Swing", "All"),
  Band       = c("Delta", "Theta", "Alpha", "Beta", "Beta_low", "Beta_high", "Gamma",
                 "Exponent", "Offset", "All")
)


