library(tidyverse)

df     <- read_csv2("Data/Behavioral Data/data_tidy_combined.csv") |>
            group_by(domain) |> 
            rename(item = ID_item) |> 
            mutate(ID_n = dense_rank(ID) - 1) |> 
            ungroup()

design_countries <-  read_csv2("Data/Behavioral Data/design_data_countries.csv") |> 
                        select(ID_item = ID, item) |> 
                        add_column(domain = "Countries")

design_food      <-  read_csv2("Data/Behavioral Data/design_data_food.csv") |> 
                        select(ID_item = ID, item) |> 
                        add_column(domain = "Food")

design_mammals   <-  read_csv2("Data/Behavioral Data/design_data_mammals.csv") |> 
                        select(ID_item = ID, item) |> 
                        add_column(domain = "Mammals")


design <- bind_rows(design_countries,design_food,design_mammals)


df <- left_join(df,design, by = c("item","domain"))


# Define instructions:
instr_food <- "Your task is to estimate the carbohydrate content (g per 100g) of 80 different food items. 
               The task consists of two phases: a training phase and a testing phase.
               Your task in the  training phase is to repeatedly estimate the carbohydrate content for the same 12 exemplar food items.
               After each estimate, you will receive feedback about the actual value of each of the 12 exemplar food items.
               Try to learn and memorise these values as best you can, since this information will help you in the testing phase.
               Your task in the testing phase is then to estimate the carbohydrate content of the remaining 68 food items as accuracte as possible.\n\n"


instr_countries <- "Your task is to estimate the life expectancy (in years) of 80 different countries. 
                    The task consists of two phases: a training phase and a testing phase.
                    Your task in the  training phase is to repeatedly estimate the life expectancy for the same 12 exemplar countries.
                    After each estimate, you will receive feedback about the actual value of each of the 12 exemplar countries.
                    Try to learn and memorise these values as best you can, since this information will help you in the testing phase.
                    Your task in the testing phase is then to estimate the life expectancy of the remaining 68 countries. as accuracte as possible.\n\n"


instr_mammals <- "Your task is to estimate the days until female maturity of 80 different mammals. 
                  The task consists of two phases: a training phase and a testing phase.
                  Your task in the  training phase is to repeatedly estimate the days until female maturity for the same 12 exemplar mammals.
                  After each estimate, you will receive feedback about the actual value of each of the 12 exemplar mammals.
                  Try to learn and memorise these values as best you can, since this information will help you in the testing phase.
                  Your task in the testing phase is then to estimate the days until female maturity of the remaining 68 mammals as accuracte as possible.\n\n"


instructions_dict <- list(
  Food      = instr_food,
  Countries = instr_countries,
  Mammals   = instr_mammals
)


# Make tidy data sets for each domain 
df <- df |>
        filter((phase == "training" | phase == "testing" & training == "0"),
               ID_item != "Basketball") |>
        select(ID, ID_n, domain, phase, block, trial, ID_item, item, training, est, true) |>
        rowwise() |> 
        mutate(instr = instructions_dict[[domain]])


temp <- df |> 
  group_by(ID,domain) |> 
  summarize(n_trials = n())


# Function to format individual trials
format_trial <- function(item, estimate, true_val, phase, domain) {

  # Format numeric values (removing trailing zeros if necessary)
  est  <- as.numeric(gsub(",", ".", estimate))
  true <- as.numeric(gsub(",", ".", true_val))
  
  phase_tag <- ifelse(phase == "training", "[TRAIN]", "[TEST]")
  
  if(domain == "Food"){
    crit_tag <- "the carbohydrate content"
  } else if (domain == "Mammals"){
    crit_tag <- "the days until female maturity"
  } else {
    crit_tag <- "the life expectancy"
  }
  
  core_text <- paste0(
    phase_tag,
    " Item: ", item, ". You say that ", crit_tag ," is <<", est, ">>."
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
narrative_data <- df %>%
  group_by(ID, ID_n, domain) %>%
  arrange(phase == "testing", block, trial) %>% # Ensure training comes before testing
  summarise(
    narrative = paste(first(instr),
      paste(mapply(format_trial, item, est, true, phase,domain), collapse = "\n"),
      sep = ""
    ),
    n_training = sum(phase == "training"),
    ID_items = paste(ID_item, collapse = ", "),
    .groups = "drop"
  ) 



# View the first participant's formatted prompt
cat(narrative_data$narrative[1])
narrative_data |> filter(domain == "Mammals") %>% pull(narrative) %>% .[1]
# Add domain
narrative_data <- narrative_data  |> arrange(domain, ID_n)


narrative_data |>
  select(text = narrative, domain, participant = ID_n, ID, n_training, ID_items) |>
  write_csv(file="Data/Preprocessed Data/narrative_data.csv")

