/*
 * CS 4804 - AI - Final Project
 * @author: Antuan Byalik
 * 
 * Monte Carlo epsilon soft policy algorithm for learning Connect-4
 */

//Initial imports
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import javax.swing.JOptionPane;

/*
 * The primary class - Constructs all supplemental objects and 
 * sets up the game, then runs indicated iterations and displays win/lose/tie rates.
 */
public class Connect4 {
	/*
	 * Some of the globally used constants and counters
	 */
	private static final double GAMMA = 0.5;
	private static final double WIN_RETURN = 1.0;
	private static final double LOSE_RETURN = -1.0;
	private static final double TIE_RETURN = 0.0;
	private static int Win_Counter = 0;
	private static int Lose_Counter = 0;
	private static int Tie_Counter = 0;
	private static int Col;
	private static boolean learning = true;

	/*
	 * main class that handles I/O, the initial policy for a blank board,
	 * and starts the algorithm
	 */
	public static void main(String args[]) throws IOException,
			ClassNotFoundException {
		//getting board size
		String input = JOptionPane
				.showInputDialog("Enter Connect-4 Grid size: Standard is 6x7");
		int row = Integer.parseInt(input.substring(0, 1));
		int col = Integer.parseInt(input.substring(2));
		Col = col;
		//System.out.println("Grid size: " + row + " " + col + " " + col / 2);

		//creating a board
		int[][] board = new int[row][col];
		board = initBoard(board);
		// print(board);

		//initial policy for empty board
		ArrayList<Double> PI = new ArrayList<Double>();
		int[] actions = new int[col];
		for (int i = 0; i < col; i++) {
			PI.add(0.05 / (col - 1));
			actions[i] = i;
		}
		PI.set(col / 2, 0.95 + (PI.get(col/2)));
		
		//creating the full spectrum policy
		HashMap<ArrayBoard, ArrayList<Double>> PI_Map = new HashMap<ArrayBoard, ArrayList<Double>>();
		ArrayBoard temple = new ArrayBoard(board);
		PI_Map.put(temple, PI); //inserting empty board policy

		//launching the algorithm
		MonteCarlo(board, PI_Map, actions);
	}

	/*
	 * monte carlo method that generates a policy after learning from games
	 */
	public static void MonteCarlo(int[][] board,
			HashMap<ArrayBoard, ArrayList<Double>> PI_Map, int[] actions)
			throws IOException, ClassNotFoundException {
		//the Q(s,a) hash map for storing seen action/state pairs
		HashMap<State, Double> Q = new HashMap<State, Double>();
		//the Q counter for how many times a state has been seen
		HashMap<State, Integer> Q_Count = new HashMap<State, Integer>();
		//high level - how many times to do a train/play combo
		for (int x = 0; x < 30; x++) {
			//check if we are trying to learn or want to play normally
			if (!learning) {
				//playing without learning 1000 games
				for (int i = 0; i < 1000; i++) {
					//reset board to empty state
					board = initBoard(board);
					//create a new episode with the play game function
					Episode e = playGame(board, PI_Map, actions);
					int winner = e.getWinner(); //checking winner and increasing counters
					if (winner == 1) {
						Win_Counter++;
					} else if (winner == 2) {
						Lose_Counter++;
					} else {
						Tie_Counter++;
					}
				}
				//printing results and resetting counters
				System.out.println("Won: " + Win_Counter + " Lost: "
						+ Lose_Counter + " Tie: " + Tie_Counter);
				System.out.println("Percent Win: " + (double) (Win_Counter)
						/ (Win_Counter + Lose_Counter));
				Win_Counter = 0;
				Lose_Counter = 0;
				Tie_Counter = 0;
				learning = true;
			} else {
				//need to learn on Q, run 1000 learning iterations
				for (int i = 0; i < 1000; i++) {
					//init board and get a new episode
					board = initBoard(board);
					Episode e = playGame(board, PI_Map, actions);
					int winner = e.getWinner(); //record winner info
					double baseReturn;
					if (winner == 1) {
						baseReturn = WIN_RETURN;
						Win_Counter++;
					} else if (winner == 2) {
						baseReturn = LOSE_RETURN;
						Lose_Counter++;
					} else {
						baseReturn = TIE_RETURN;
						Tie_Counter++;
					}
					//create a list of boards to hold affected states for policy change
					ArrayList<ArrayBoard> stateHelper = new ArrayList<ArrayBoard>();
					for (int j = 0; j < e.getState().size(); j++) {
						//iterating through state/action pair chain for current episode
						//adding information to state only list for policy change later
						ArrayBoard tempArrayBoard = new ArrayBoard(e.getState()
								.get(j).getBoard());
						stateHelper.add(tempArrayBoard);
						//if we've seen this state, need to average in new value
						if (Q.containsKey((e.getState().get(j)))) {
							//using the return for whatever this games result was
							//replace the current value at our current state/action pair
							//with the current return multiplied by the weighted 
							//existing return and update counter...umad
							Q.put(e.getState().get(j),
									((Q_Count.get(e.getState().get(j)) * Q
											.get(e.getState().get(j))) + baseReturn
											* (Math.pow(GAMMA, e.getState()
													.size() - j)))
											/ Q_Count.get(e.getState().get(j))
											+ 1);
							Q_Count.put(e.getState().get(j),
									Q_Count.get(e.getState().get(j)) + 1);
						} else {
							//new state, simply add in to hash map and update counter
							Q.put(e.getState().get(j),
									baseReturn
											* (Math.pow(GAMMA, e.getState()
													.size() - j)));
							Q_Count.put(e.getState().get(j), 1);
						}
					}
					// update the effected PI's
					State tempState;
					double max; //local max for update
					int act = -1;
					//iterate over list of affected states
					for (int j = 0; j < stateHelper.size(); j++) {
						max = 0.0;
						//iterate over my possible actions
						for (int k = 0; k < actions.length; k++) {
							tempState = new State(
									stateHelper.get(j).getBoard(), k);
							//if i haven't seen this action/state pair...add it with uniform probability
							if (!PI_Map.containsKey(stateHelper.get(j))) {
								ArrayList<Double> tempy = new ArrayList<Double>();
								for (int m = 0; m < board[0].length; m++) {
									tempy.add(1.0 / (board[0].length));
								}
								PI_Map.put(stateHelper.get(j), tempy);
							}
							//if i've seen this action/state pair see if its bigger than my running max
							//if so, update
							if (Q.containsKey(tempState)) {
								if (Q.get(tempState) >= max) {
									max = Q.get(tempState);
									act = k;
								}
							}
						}
						//call the update function with the new max
						PI_Map = updatePI(PI_Map,
								(act == -1) ? actions.length / 2 : act,
								stateHelper.get(j));
					}
				}
				learning = false; //reset learning off
//				System.out.println("Won: " + Win_Counter + " Lost: "
//						+ Lose_Counter + " Tie: " + Tie_Counter);
//				System.out.println("Percent Win: " + (double) (Win_Counter)
//						/ (Win_Counter + Lose_Counter));
				Win_Counter = 0;
				Lose_Counter = 0;
				Tie_Counter = 0;
			}
		}
	}

	/*
	 * update pi function that sets a particular policy based on epsilon for a state/action set pair
	 */
	public static HashMap<ArrayBoard, ArrayList<Double>> updatePI(
			HashMap<ArrayBoard, ArrayList<Double>> PI_Map, int act,
			ArrayBoard board) {
		int size = PI_Map.get(board).size();
		ArrayList<Double> tempy = PI_Map.get(board);
		for (int i = 0; i < size; i++) {
			tempy.set(i, 0.05 / size);
		}
		tempy.set(act, (0.95 + tempy.get(act)));
		PI_Map.put(board, tempy);
		return PI_Map;
	}

	/*
	 * play game method that handles connect-4 game logic 
	 */
	public static Episode playGame(int[][] board,
			HashMap<ArrayBoard, ArrayList<Double>> PI_Map, int[] actions) {
		int move = -1;
		int player = 1;
		ArrayList<State> stateList = new ArrayList<State>();
		//play until game is over by either player winning or a tie
		while (true) {
			// print(board);
			//if player 1 use policy, otherwise use random move
			if (player == 1) {
				move = generateMove(PI_Map, board);
			} else {
				move = randomMove(board);
			}
			//flip these two to test random vs monte and monte vs monte 
			//move = generateMove(PI_Map, board);
			if (move == -1) {
				Episode e = new Episode(stateList, 0);
				return e;
			}
			//take the move and physically insert into array
			board = execMove(board, move, player);
			//check for either player win or a tie
			if (checkWin(board, move)) {
				Episode e = new Episode(stateList, player);
				return e;
			}
			if (checkTie(board)) {
				Episode e = new Episode(stateList, 0);
				return e;
			}
			//flip whose turn it is and record state/action if player 1
			if (player == 1) {
				State state = new State(board, move);
				stateList.add(state);
				player = 2;
			} else {
				player = 1;
			}
		}
	}

	/*
	 * quick random move maker for random opponent strategy
	 */
	public static int randomMove(int[][] board) {
		Random rand = new Random();
		int move;
		//while a valid move isnt made
		while (true) {
			//make sure board isn't full
			if (checkTie(board)) {
				return -1;
			}
			//get move and see if it can be placed
			move = rand.nextInt(Col);
			for (int i = board.length - 1; i >= 0; i--) {
				if (board[i][move] == 0) {
					return move;
				}
			}
		}
	}

	/*
	 * see if board is full
	 */
	public static boolean checkTie(int[][] board) {
		for (int i = 0; i < board.length; i++) {
			//only need to check top row
			if (board[0][i] == 0) {
				return false;
			}
		}
		return true;
	}

	/*
	 * check for a win
	 */
	public static boolean checkWin(int[][] board, int move) {
		int row = -1;
		int player = -1;
		//getting the last move's row as we only have column
		for (int i = 0; i < board.length; i++) {
			if (board[i][move] != 0) {
				row = i;
				player = board[i][move];
			}
		}
		//checking horizontal win
		int count = 0;
		int i = row;
		int j = move;
		//iterate right
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i + 1, j)) {
				break;
			}
			i++;
		}
		i = row;
		j = move;
		//iterate left
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i - 1, j)) {
				break;
			}
			i--;
		}
		//5 as we double counted the last move 
		if (count >= 5) {
			return true; //win 
		}

		//same as above for vertical
		count = 0;
		i = row;
		j = move;
		//iterate up
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i, j + 1)) {
				break;
			}
			j++;
		}
		i = row;
		j = move;
		//iterate down
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i, j - 1)) {
				break;
			}
			j--;
		}
		if (count >= 5) {
			return true; //win
		}

		//diagonal 1 check
		count = 0;
		i = row;
		j = move;
		//iterate up/right
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i + 1, j + 1)) {
				break;
			}
			i++;
			j++;
		}
		i = row;
		j = move;
		//iterate down/left
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i - 1, j - 1)) {
				break;
			}
			i--;
			j--;
		}
		if (count >= 5) {
			return true; //win
		}

		//other diagonal check
		count = 0;
		i = row;
		j = move;
		//iterate left/up
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i - 1, j + 1)) {
				break;
			}
			i--;
			j++;
		}
		i = row;
		j = move;
		//iterate right/down
		while (board[i][j] == player) {
			count++;
			if (!validCheck(board, i + 1, j - 1)) {
				break;
			}
			i++;
			j--;
		}
		if (count >= 5) {
			return true; //win
		}

		// no win
		return false;
	}

	/*
	 * normal check for inside physical board
	 */
	public static boolean validCheck(int[][] board, int row, int col) {
		if (row < 0 || row > board.length - 1) {
			return false;
		}
		if (col < 0 || col > board[0].length - 1) {
			return false;
		}
		return true; //all good
	}

	/*
	 * insert move into board, all checks have already occured
	 */
	public static int[][] execMove(int[][] board, int move, int player) {
		for (int i = board.length - 1; i >= 0; i--) {
			if (board[i][move] == 0) {
				board[i][move] = player;
				return board;
			}
		}
		//safety check...dem null pointers
		System.out.println("Error in board execMove");
		return board;
	}

	/*
	 * sum up the policy's in a given state to action set mapping
	 */
	public static double sumPI(HashMap<ArrayBoard, ArrayList<Double>> PI_Map,
			ArrayBoard board) {
		//this method just makes sure there are playable moves at a given state
		double sum = 0;
		for (double i : PI_Map.get(board)) {
			sum += i;
		}
		return sum;
	}

	/*
	 * use the policy to produce a valid move
	 */
	public static int generateMove(
			HashMap<ArrayBoard, ArrayList<Double>> PI_Map, int[][] board) {
		ArrayBoard temple = new ArrayBoard(board);
		Random rand = new Random();
		//get a random double between 0.0 and 1
		double temp = rand.nextDouble();
		//if this is first time seeing a state/action pair
		//insert skewed distribution as policy
		if (!PI_Map.containsKey(temple)) {
			ArrayList<Double> tempy = new ArrayList<Double>();
			for (int i = 0; i < board[0].length; i++) {
				tempy.add(0.05 / (board[0].length));
			}
			tempy.set(board[0].length / 2,
					(tempy.get(board[0].length / 2)) + .95);
			PI_Map.put(temple, tempy);
		}
		//use the val from the move to see where we fall into the probability
		double val = PI_Map.get(temple).get(0);
		//if less than first cell's value, use this move
		if (temp < val) {
			//if valid, just return it
			if (validMove(board, 0)) {
				return 0;
			} else {
				//otherwise this column is full 
				ArrayList<Double> tempy = PI_Map.get(temple);
				for (int i = 1; i < tempy.size(); i++) {
					tempy.set(i, tempy.get(i) + tempy.get(0)
							/ (tempy.size() - 1));
				}
				//temp set it to be zero so its not generated again this episode
				tempy.set(0, 0.0);
				PI_Map.put(temple, tempy);
				//while we have valid moves from this state, generate a new one and try again
				if (sumPI(PI_Map, temple) > 0.0) {
					Random randy = new Random();
					int nextMove;
					while (true) {
						nextMove = randy.nextInt(Col);
						if (validMove(board, nextMove)) {
							return nextMove;
						}
						//check tie so we dont' loop inf
						if (checkTie(board)) {
							return -1;
						}
					}
					// generateMove(PI_Map, board);
				} else {
					return -1;
				}
			}
		}
		//iterative case for the rest of the action set excluding the last one
		for (int i = 0; i < PI_Map.get(temple).size() - 1; i++) {
			//if we're in the range of this value use it if its valid
			if (temp >= PI_Map.get(temple).get(i)
					&& (temp < (val + PI_Map.get(temple).get(i + 1)))) {
				if (validMove(board, i + 1)) {
					return i + 1; //return if valid
				} else {
					//column full have to repeat procedure above
					ArrayList<Double> tempy = PI_Map.get(temple);
					for (int j = 0; j < tempy.size(); j++) {
						if (j != i + 1) {
							tempy.set(j, tempy.get(j) + tempy.get(i + 1)
									/ (tempy.size() - 1));
						}
					}
					tempy.set(i + 1, 0.0);
					PI_Map.put(temple, tempy);
					if (sumPI(PI_Map, temple) > 0.0) {
						Random randy = new Random();
						int nextMove;
						while (true) {
							//loop until board full or valid move found
							nextMove = randy.nextInt(Col);
							if (validMove(board, nextMove)) {
								return nextMove;
							}
							if (checkTie(board)) {
								return -1;
							}
						}
						// generateMove(PI_Map, board);
					} else {
						return -1; //no valid moves left at this board
					}
				}
			}
			val += PI_Map.get(temple).get(i + 1); //sum as we go along to find probabilty
		}
		//same as above case for last cell
		if (validMove(board, board[0].length - 1)) {
			return board[0].length - 1; //return if valid
		} else {
			//no room in column
			ArrayList<Double> tempy = PI_Map.get(temple);
			for (int i = 1; i < tempy.size(); i++) {
				tempy.set(i, tempy.get(i) + tempy.get(tempy.size() - 1)
						/ (tempy.size() - 1));
			}
			tempy.set(tempy.size(), 0.0);
			PI_Map.put(temple, tempy);
			if (sumPI(PI_Map, temple) > 0.0) {
				Random randy = new Random();
				int nextMove;
				while (true) {
					//keep going until move found or board full
					nextMove = randy.nextInt(Col);
					if (validMove(board, nextMove)) {
						return nextMove;
					}
					if (checkTie(board)) {
						return -1;
					}
				}
				// generateMove(PI_Map, board);
			} else {
				return -1; //board full
			}
		}
	}

	/*
	 * checks to see if a move overflows a column
	 */
	public static boolean validMove(int[][] board, int val) {
		for (int i = board.length - 1; i >= 0; i--) {
			if (board[i][val] == 0) {
				return true;
			}
		}
		return false; //no room left
	}

	//initializes an empty board to all 0's 
	public static int[][] initBoard(int[][] board) {
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				board[i][j] = 0;
			}
		}
		return board;
	}

	/*
	 * prints the board
	 */
	public static void print(int[][] board) {
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				System.out.print(board[i][j]);
			}
			System.out.println();
		}
	}

	/*
	 * private nested class representing an episode
	 * wraps a State object list to a int for the winner of the game
	 */
	private static class Episode {
		private ArrayList<State> state; //state list
		private int winner; //int for the winner 1 for p1 and 2 for p2

		//constructor
		public Episode(ArrayList<State> state, int winner) {
			this.state = state;
			this.winner = winner;
		}

		public ArrayList<State> getState() {
			return state;
		}

		public int getWinner() {
			return winner;
		}
	}

	/*
	 * private nested state class that represents an action and a board
	 * used to wrap the current board and the corresponding action from the action
	 * set taken at that point
	 */
	private static class State {
		private int[][] Board;
		private int action;

		/*
		 * constructor to initialize it
		 */
		public State(int[][] Board, int action) {
			this.Board = Board;
			this.action = action;
		}

		public int[][] getBoard() {
			return Board;
		}

		public int getAction() {
			return action;
		}

		public void setRow(int action) {
			this.action = action;
		}
	}

	/*
	 * private nested class for the array board 
	 * used as a wrapper so that the array can be put into hashmaps
	 */
	private static class ArrayBoard {
		private int[][] Board;

		/*
		 * constructor to initialize it
		 */
		public ArrayBoard(int[][] Board) {
			this.Board = Board;
		}

		public int[][] getBoard() {
			return Board;
		}
	}
}
