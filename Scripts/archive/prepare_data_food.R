library(tidyverse)

domain <- "Food" # "Mammals" "Countries"
df     <- read_csv2("Data/Behavioral Data/data_tidy_combined.csv")

# Make tidy data sets for each domain 
df_food <- df |>
  filter(domain == "Food",
         (phase == "training" | phase == "testing" & training == "0"),
         ID_item != "Basketball") |>
  select(ID, phase, block, trial, ID_item, training, est, true)

n <- df_food$ID |> n_distinct()

# 2. Define the header/instructions based on your task info
instructions <- "Your task is to estimate the carbohydrate content (g per 100g) of 80 different food items. 
                 The task consists of two phases: a training phase and a testing phase.
                 Your task in the  training phase is to repeatedly estimate the carbohydrate content for the same 12 exemplar food items.
                 After each estimate, you will receive feedback about the actual value of each of the 12 exemplar food items.
                 Try to learn and memorise these values as best you can, since this information will help you in the testing phase.
                 
                 Your task in the testing phase is then to estimate the carbohydrate content of the remaining 68 food items as accuracte as possible.\n\n"

# 3. Function to format individual trials
format_trial <- function(item, estimate, true_val, phase) {
  # Format numeric values (removing trailing zeros if necessary)
  est  <- as.numeric(gsub(",", ".", estimate))
  true <- as.numeric(gsub(",", ".", true_val))
  
  phase_tag <- ifelse(phase == "training", "[TRAIN]", "[TEST]")
  core_text <- paste0(
    phase_tag,
    " Item: ", item, ". You say that the carbohydrate content is <<", est, ">>."
  )
  
  # Only training phase includes feedback in the prompt
  if (phase == "training") {
    
    feedback <- paste0(" The correct answer for ", item ," is ", true)
    
    return(paste0(core_text, feedback))
  } else {
    return(core_text)
  }
}

# 4. Process data into narratives
narrative_data <- df_food %>%
  group_by(ID) %>%
  arrange(phase == "testing", block, trial) %>% # Ensure training comes before testing
  summarise(
    narrative = paste(
      instructions,
      paste(mapply(format_trial, ID_item, est, true, phase), collapse = "\n"),
      sep = ""
    )
  )

# View the first participant's formatted prompt
cat(narrative_data$narrative[1])

# Add domain
narrative_data <- narrative_data |>
                    add_column(domain="Food",
                               IDn   = (1:n)-1)


narrative_data |>
  select(text = narrative, domain, participant = IDn) |>
  write_csv(file="Data/Preprocessed Data/narrative_data_food.csv")


# Split into train and test
# set.seed(17824)
# 
# IDs          <- narrative_data$ID
# n_train      <- round(length(IDs)*0.8)
# training_IDs <- sample(IDs,n_train)
# testing_IDs  <- IDs[!IDs %in% training_IDs]
# 
# # Save csv
# 
# narrative_data |> 
#   filter(IDs %in% training_IDs) |> 
#   select(text = narrative, domain, participant = IDn) |> 
#   write_csv(, file="Data/Preprocessed Data/TRAIN_narrative_data_food.csv")
# 
# narrative_data |> 
#   filter(IDs %in% testing_IDs) |> 
#   select(text = narrative, domain, participant = IDn) |> 
#   write_csv(, file="Data/Preprocessed Data/TEST_narrative_data_food.csv")
