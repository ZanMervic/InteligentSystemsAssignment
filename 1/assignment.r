library(GA)

map_reading <- function(maze){
  #In mazer.r all mazes are square shaped so we might only need the rows parameter
  rows = length(maze)
  columns = nchar(maze[1])
  #We initialize an empty matrix of the correct dimensions
  maze_matrix = matrix(0, nrow = rows, ncol = columns)
  #We fill the matrix with correct values
  for (i in c(1:rows)) {
    maze_matrix[i,] <- unlist(strsplit(maze[i], ""))
  }
  return (maze_matrix)
}

#FOR TESTING ONLY
test_solution <- function(path){
  maze_matrix = map_reading(my_maze)
  start <- as.vector(t(which(maze_matrix == 'S', arr.ind = TRUE)))
  exit <- as.vector(t(which(maze_matrix == 'E', arr.ind = TRUE)))
  current = start
  penalties <- 0.0
  treasures <- 0
  for (i in seq(1,length(path), 2)) {
    move <- path[i : (i+1)]
    if(all(move == c(0,0))){ #UP
      current <- current - c(1,0)
    } else if(all(move == c(0,1))){ #RIGHT
      current <- current + c(0,1)
    } else if(all(move == c(1,0))){ #DOWN
      current <- current + c(1,0)
    } else if(all(move == c(1,1))){ #LEFT
      current <- current - c(0,1)
    } else{
      print("Error")
    }
    if(!all(current > 0) || !all(current <= dim(maze_matrix)) || maze_matrix[current[1], current[2]] == '#'){
      print("HIT A WALL")
      return(list(maze=maze_matrix, position=current, penalties=penalties, treasures=treasures))
    }
    if(maze_matrix[current[1], current[2]] == 'o' || maze_matrix[current[1], current[2]] == 'S'){
      penalties <- penalties + 10
    }else{
      penalties <- penalties - 5
    }
    if(maze_matrix[current[1], current[2]] == 'T'){
      treasures <- treasures + 1
    }
    if(maze_matrix[current[1], current[2]] == 'E'){
      print("MADE IT TO THE END")
      return(list(maze=maze_matrix, position=current, penalties=penalties, treasures=treasures))
    }
    maze_matrix[current[1], current[2]] <- "o"
  }
  print("RAN OUT OF MOVES")
  return(list(maze=maze_matrix, position=current, penalties=penalties, treasures=treasures))
  
}



#Function that gets a maze, move and current position and changes the move and position to a valid one
#We need to make sure we only made 1 invalid move And that we are not out of bounds!!! 
#Possible improvements, recieve the matrix with a "o" where we have been and prioritising moves that dont get us on a "o" but rather on a "."
generate_valid_move <- function(maze_matrix, prev_move, current){
  
    #We change the position to the one before the move
    if(all(prev_move == c(0,0))){ #UP -> go down
      current <- current + c(1,0)
    } else if(all(prev_move == c(0,1))){ #RIGHT -> go left
      current <- current - c(0,1)
    } else if(all(prev_move == c(1,0))){ #DOWN -> go up
      current <- current - c(1,0)
    } else if(all(prev_move == c(1,1))){ #LEFT -> go right
      current <- current + c(0,1)
    } else{
      print("Error in change_to_valid")
    }
    
    
    #List of possible moves
    possible_moves <- matrix(c(0,0,0,1,1,0,1,1), ncol = 2, byrow = TRUE)
    count = 4
    
    #We chech all neighboring cells for walls, and remove invalid possible moves
    if(current[1] <= 1 || maze_matrix[current[1] - 1, current[2]] == "#"){
      possible_moves <- possible_moves[-(count-3),]
      count <- count - 1
    }
    if(current[2] <= 1 || maze_matrix[current[1], current[2] + 1] == "#"){
      possible_moves <- possible_moves[-(count-2),]
      count <- count - 1
    }
    if(current[1] >= dim(maze_matrix)[1] || maze_matrix[current[1] + 1, current[2]] == "#"){
      possible_moves <- possible_moves[-(count-1),]
      count <- count - 1
    }
    if(current[2] >= dim(maze_matrix)[2] || maze_matrix[current[1], current[2] - 1] == "#"){
      possible_moves <- possible_moves[-count,]
      count <- count - 1
    }
    
    
    #We select a random move from remaining possible moves
    if(is.null(dim(possible_moves)[1])){
      new_move <- possible_moves
    } else{
      new_move <- possible_moves[sample(1:dim(possible_moves)[1], 1),]
    }
    
    
    #We adjust the current position accordingly
    if(all(new_move == c(0,0))){ #UP
      current <- current - c(1,0)
    } else if(all(new_move == c(0,1))){ #RIGHT
      current <- current + c(0,1)
    } else if(all(new_move == c(1,0))){ #DOWN
      current <- current + c(1,0)
    } else if(all(new_move == c(1,1))){ #LEFT
      current <- current - c(0,1)
    } else{
      print("Error2 in change_to_valid")
    }
    
    return(matrix(c(current, new_move), ncol = 2, byrow = TRUE))
  
}



custom_population <- function(object,...){
  #We create a starting random matrix of our population
  my_population <- matrix(sample(0:1, my_nBits * my_popSize, replace = TRUE), nrow=my_popSize)
  
  #We set the minimum required steps for our entity to be valid
  required_steps <- my_required_steps
  
  #We check how many steps are made before going into a wall, if its less than required steps we generate a new value
  maze_matrix = map_reading(my_maze)
  for(row in 1:my_starting_population_size){
    path <- my_population[row,]
    steps_made <- 0
    
    #We count the moves with our current path and change the path if invalid
    current <- as.vector(t(which(maze_matrix == 'S', arr.ind = TRUE)))
    for (i in seq(1, length(path), 2)) {
      move <- path[i : (i+1)]
      if(all(move == c(0,0))){ #UP
        current <- current - c(1,0)
      } else if(all(move == c(0,1))){ #RIGHT
        current <- current + c(0,1)
      } else if(all(move == c(1,0))){ #DOWN
        current <- current + c(1,0)
      } else if(all(move == c(1,1))){ #LEFT
        current <- current - c(0,1)
      } else{
        print('Error in custom pupulation')
      }
      
      #If we hit a wall we generate a different (valid) move
      if(!all(current > 0) || !all(current <= dim(maze_matrix)) || maze_matrix[current[1], current[2]] == '#'){
        temp <- generate_valid_move(maze_matrix, move, current)
        move <- temp[2,]
        path[i:(i+1)] <- move
        current <- temp[1,]
        if(!all(current > 0) || !all(current <= dim(maze_matrix)) || maze_matrix[current[1], current[2]] == '#'){
          print("Something went wrong, aborting")
          break
        }
      }
      
      if(maze_matrix[current[1], current[2]] == 'E'){
        #If we have a solution that reaches the finish we keep it
        break
      }
      
      steps_made <- steps_made + 1
      
      #If we made enough steps we stop the iteration
      if(steps_made >= required_steps){
        break
      }
    }
    
    my_population[row,] <- path
    steps_made = 0
  }
  return(my_population)
}


custom_mutation <- function(object, parent)
{
  #Select the parent vector from the population
  mutate <- parent <- as.vector(object@population[parent,])
  n <- length(parent)
  #Sample a random vector element that should be changed
  #We make sure j is a odd number (first number of a move pair)
  j <- sample(1:n, size = 1)
  if (j %% 2 != 1){
    j <- j -1
  }
  
  #We now go from the first to the j-th bit and check if there is a wall hit, if there
  #is we make a mutation on the bit responsible for the hit
  maze_matrix = map_reading(my_maze)
  current <- as.vector(t(which(maze_matrix == 'S', arr.ind = TRUE)))
  move <- c(0,0)
  for (i in seq(1, j, 2)) {
    move <- mutate[i : (i+1)]
    if(all(move == c(0,0))){ #UP
      current <- current - c(1,0)
    } else if(all(move == c(0,1))){ #RIGHT
      current <- current + c(0,1)
    } else if(all(move == c(1,0))){ #DOWN
      current <- current + c(1,0)
    } else if(all(move == c(1,1))){ #LEFT
      current <- current - c(0,1)
    } else{
      print("Error in custom mutation")
    }
    if(!all(current > 0) || !all(current < dim(maze_matrix)) || maze_matrix[current[1], current[2]] == '#'){
      #If we hit a wall before we get to the randomly selected "j" bit we change j to the current bit
      #so we make the mutation where we would have hit the wall
      j = i
      break
    }
    if(maze_matrix[current[1], current[2]] == 'E'){
      return(mutate)
    }
  }
  
  
  #Change that element to a random move that is different than the invalid one
  new_move <- generate_valid_move(maze_matrix, move, current)[2,]
  
  'new_move <- c(sample(0:1, 1),sample(0:1, 1))
  while(all(new_move == move)){
    new_move <- c(sample(0:1, 1),sample(0:1, 1))
  }'
  
  
  mutate[j] <- new_move[1]
  mutate[j + 1] <- new_move[2]
  return(mutate)
}



fitness <- function(path){
  #Dobimo vektor koordinat zacetne pozicije, inicializiramo ostale spremenljivke
  maze_matrix = map_reading(my_maze)
  start <- as.vector(t(which(maze_matrix == 'S', arr.ind = TRUE)))
  exit <- as.vector(t(which(maze_matrix == 'E', arr.ind = TRUE)))
  current = start
  width = length(maze_matrix[1,])
  penalties <- 0.0
  
  
  
  #Gremo skozi celoten vektor (po 2 bita pretvorimo v move) in spreminjamo trenutno lokacijo, ter preverjamo veljavnost koraka
  for (i in seq(1,length(path), 2)) {
    #We get the next move
    move <- path[i : (i+1)]
    #We adjust the current position accordingly
    if(all(move == c(0,0))){ #UP"
      current <- current - c(1,0)
    } else if(all(move == c(0,1))){ #RIGHT
      current <- current + c(0,1)
    } else if(all(move == c(1,0))){ #DOWN
      current <- current + c(1,0)
    } else if(all(move == c(1,1))){ #LEFT
      current <- current - c(0,1)
    } else{
      print("Error in fitness")
    }
    
    
    
    
    if(!all(current > 0) || !all(current <= dim(maze_matrix)) || maze_matrix[current[1], current[2]] == '#'){
      #We are either out of a matrix or on a wall, dont return 0 because its better if we 
      #hit the wall closer (and make more valid moves) to the exit so we should reward that
      #If we add -1000 in front it will work better for some mazes but worse at other
      return (-penalties - 2 * sqrt((current[1] - exit[1])^2 + (current[2] - exit[2])^2))
    }
    
    if(maze_matrix[current[1], current[2]] == 'o' || maze_matrix[current[1], current[2]] == 'S'){
      #If we go to an already visited spot we get a penalty
      penalties <- penalties + 10
    }else{
      #If we go to a unvisited spot we get rewarded
      penalties <- penalties - 5
    }
    
    if(maze_matrix[current[1], current[2]] == 'T'){
      #If we collect a treasure we get rewarded and the visited tiles reset
      penalties <- penalties - 200
      maze_matrix[maze_matrix == 'o'] <- '-'
      maze_matrix[maze_matrix == 'S'] <- 's'
    }
    
    if(maze_matrix[current[1], current[2]] == 'E'){
      #We made it to the exit, we reward the path with the least steps
      return (1000 + 1000/i - min(penalties, 0))
    }
    
    #We mark the tile as visited, so we dont visit it again
    maze_matrix[current[1], current[2]] <- "o"
    
    
  }
  #All moves were done and we did not make it to the exit
  return(100/sqrt((current[1] - exit[1])^2 + (current[2] - exit[2])^2) - penalties)
}

#ALL SETTINGS----------------------------------------

my_maze <- maze5 #MAZE

#ga settings
my_nBits <- ceiling((nchar(my_maze[1])^2) / 2) * 2 #number of bits -> A suitable value for nBits is (nchar(mazex[1])^2) / 2
my_popSize = 1000 #population size
my_maxiter = 1000 #number of iterations
my_run = 50 #number of runs
my_pmutation = 0.3 #mutation chance

#starting population settings -best for normal mazes
my_required_steps = floor(my_nBits / 2) #number of valid steps first gen entities need to make -> MAX my_nbits / 2 !!!
my_starting_population_size = my_popSize #number of "trained" first gen entities -> MAX my_popSize, min = 1 (gives a random population)!!!

#best for treasure mazes
my_required_steps = max((my_nBits / 2), 20) #A low value is best for treasure mazes because it doesn't allow entities to reach the end thus getting many points and missing a treasure
my_starting_population_size = 1 #number of "trained" first gen entities -> MAX my_popSize, min = 1 (gives a random population)!!!


#----------------------------------------------------

GA <- ga(type = "binary", fitness = fitness, nBits = my_nBits, maxiter = my_maxiter, run = my_run, popSize = my_popSize, pmutation = my_pmutation, mutation=custom_mutation, population = custom_population)


summary(GA)
GA@solution

test_solution(GA@solution[1,])

#RESULTS ----------------------------------------------

#Can complete mazes consistently: 1,2,3,4,5,6, 3T,4T,5T
#Can collect all treasures: 3T,4T,(5T),