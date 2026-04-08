library(tidyverse)

df <- read_csv2("Data/data_tidy_combined.csv")


# Make tidy data sets for each domain 
df_food <- df |>
            filter(domain == "Food") |> 
            select(ID,phase,block,trial,ID_item,training,true,true)


# 2. Define the header/instructions based on your task info
instructions <- "Your task is to estimate the carbohydrate content per 100g of 80 food items. 
                 The task consists of two phases: a training phase and a testing phase.
                 
                 During the training phase, you will see a total of 12 different foods. 
                 Your task in the  training phase is to estimate the carbohydrate content per 100g for each of the 12 foods.
                 After each judgment, you will receive feedback about the actual value. 
                 Try to learn and memorise these values as best you can, since this information will help you in the testing phase.
                 
                 Your task in the testing phase is then to estimate the carbohydrate content
                 per 100g of all 80 food items.  There will be no feedback in the testing phase.
                 
                 How many g carbohydrates per 100g does this food item have?\n\n"

# 3. Function to format individual trials
format_trial <- function(item, estimate, true_val, phase) {
  # Format numeric values (removing trailing zeros if necessary)
  est  <- as.numeric(gsub(",", ".", estimate))
  true <- as.numeric(gsub(",", ".", true_val))
  
  core_text <- paste0("Item: ", item, ". You say that the carbohydrate content is <<", est, ">>.")
  
  # Only training phase includes feedback in the prompt
  if (phase == "training") {
    
    feedback <- paste0(" That food item:", item ," has ", true, "g carbohydrates per 100g.")
    
    return(paste0(core_text, feedback))
  } else {
    return(core_text)
  }
}

# 4. Process data into narratives
narrative_data <- df %>%
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
narrative_data_food <- narrative_data |> add_column(domain="Food")

# Save csv
write_csv2(narrative_data_food, file="Data/narrative_data_food.csv")
