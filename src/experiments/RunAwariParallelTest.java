package experiments;

import game.Game;
import main.Constants;
import other.RankUtils;
import mcts.PNSMCTS_L2;
import mcts.PNSMCTS_L2_MAST;
import mcts.PNSMCTS_L2_RAVE;
import mcts.PNSMCTS_MAST;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.move.Move;
import other.trial.Trial;
import search.mcts.MCTS;
import search.minimax.AlphaBetaSearch;
import utils.Utils;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.Scanner;

// AI types that can be used in the comparison
enum AIType {
    PNSMCTS_L2_MAST,
    PNSMCTS_MAST,
    PNSMCTS_L2_RAVE,
    PNSMCTS_RAVE,
    PNSMCTS_L2,
    MCTS,
    ALPHA_BETA
}

/**
 * Compares different AI players against each other in a configurable way.
 * Supports PNSMCTS variants, MCTS, and Alpha-Beta search.
 */
public class RunAwariParallelTest {

    // Available game files
    private static final String[] GAME_FILES = {
//            "games/Awari.lud"
            "games/Knightthrough.lud"
//            "games/Minishogi.lud"
//            "games/Lines of Action 8x8.lud"
//            "games/Lines of Action 7x7.lud"
//            "board/space/blocking/Amazons.lud"
//            "board/war/replacement/checkmate/shogi/Shogi.lud"
//            "board/war/replacement/checkmate/chess/Chess.lud"
    };

    private static final String[] GAME_NAMES = {
//            "Awari"
            "Knightthrough"
//            "Minishogi"
//            "Lines of Action 8x8"
//            "Lines of Action 7x7"
//            "Amazons"
//            "Shogi"
//            "Chess"
    };

    // Game configuration
    private static final int NUM_GAMES = 1000; // Number of games per instance
    private static final double TIME_PER_MOVE = 1.0; // Time per move in seconds
//    private static final int MAX_ITERATIONS = 50000; // Recommended starting point for MCTS/PNSMCTS strength
//    private static final int MAX_DEPTH = 500; // Large depth to prevent pruning of long games

    // Thread pool configuration - 4 parallel instances per game type
    private static final int MAX_CORES = 15;  // Maximum number of cores to use
    private static final int AVAILABLE_CORES = Math.min(MAX_CORES, Runtime.getRuntime().availableProcessors());
    // Number of parallel instances per game type
    // Reduced to 8 to prevent resource exhaustion on a 10-core system
    private static final int INSTANCES_PER_GAME = 15;
    // Total number of game types
    private static final int NUM_GAME_TYPES = GAME_FILES.length;
    // Total parallel instances (4 per game type)
    private static final int GAMES_IN_PARALLEL = NUM_GAME_TYPES * INSTANCES_PER_GAME;  // 5 * 4 = 20 instances
    // Calculate cores per game instance
    private static final int USABLE_CORES = Math.max(1, AVAILABLE_CORES / 2);  // Use half of available cores for system

    // Print system info for debugging
    static {
        System.out.println("System Info - Available CPU Cores: " + AVAILABLE_CORES);
        System.out.println("Using " + GAMES_IN_PARALLEL + " parallel game instances");
        System.out.println("Leaving " + (AVAILABLE_CORES - USABLE_CORES) + " cores free for system/other applications");
    }
    private static final Scanner scanner = new Scanner(System.in);

    // AI Configuration - Change these to test different AI matchups
    private static final AIType PLAYER1_AI = AIType.PNSMCTS_L2_RAVE;
    private static final AIType PLAYER2_AI = AIType.ALPHA_BETA;

    // AI parameters (if needed)
    // MCTS and Alpha-Beta will use their default parameters

    /**
     * Creates an AI instance of the specified type
     */
    private static AI createAI(AIType type, Game game, int playerId) {
        // params to test:
        boolean finMove = true;
        int minVisits = 5;
        double pnCons = 1.0;
        double cFactor = 0.2;
        switch (type) {
            case PNSMCTS_L2_MAST: {
                // Initialize with the specified player ID
                PNSMCTS_L2_MAST ai = new PNSMCTS_L2_MAST(finMove, minVisits, pnCons, cFactor, 1);
                ai.initAI(game, playerId);
                return ai;
            }
            case PNSMCTS_MAST: {
                // Initialize with the specified player ID
                PNSMCTS_MAST ai = new PNSMCTS_MAST(finMove, minVisits, pnCons, 1);
                ai.initAI(game, playerId);
                return ai;
            }
            case PNSMCTS_L2_RAVE: {
                //double[] raveSettings = {1, Math.sqrt(2), 1};
//                PNSMCTS_L2_RAVE ai = new PNSMCTS_L2_RAVE(raveSettings);
                PNSMCTS_L2_RAVE ai = new PNSMCTS_L2_RAVE(finMove, minVisits, pnCons, cFactor);
                ai.initAI(game, playerId);
                return ai;
            }
            case PNSMCTS_L2: {
                PNSMCTS_L2 ai = new PNSMCTS_L2(finMove, minVisits, pnCons, cFactor);
                ai.initAI(game, playerId);
                return ai;
            }
            case MCTS:
                // Create MCTS with standard UCT settings
                return MCTS.createUCT();
            case ALPHA_BETA:
                // Create Alpha-Beta with standard settings
                return new AlphaBetaSearch();
            default:
                throw new IllegalArgumentException("Unknown AI type: " + type);
        }
    }

    /**
     * Gets the name of an AI type
     */
    private static String getAIName(AIType type) {
        switch (type) {
            case PNSMCTS_L2_MAST: return "PNSMCTS_L2_MAST";
            case PNSMCTS_MAST: return "PNSMCTS_MAST";
            case PNSMCTS_L2_RAVE: return "PNSMCTS_L2_RAVE";
            case PNSMCTS_L2: return "PNSMCTS_L2";
            case MCTS: return "MCTS(Standard UCT)";
            case ALPHA_BETA: return "Alpha-Beta(Standard)";
            default: return type.name();
        }
    }

    public static void main(final String[] args) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("            AI COMPARISON TOOL - " + PLAYER1_AI + " vs " + PLAYER2_AI);
        System.out.println("=".repeat(70));
        System.out.println("Running " + GAME_FILES.length + " different game types in parallel");
        System.out.println("Each game type will run " + NUM_GAMES + " matches");
        System.out.println("Time per move: " + TIME_PER_MOVE + "s");
        System.out.println("=".repeat(70) + "\n");

        // Create a thread pool with optimal size for the system
        ExecutorService executor = Executors.newFixedThreadPool(GAMES_IN_PARALLEL);//max_cores
        System.out.println("Starting " + GAMES_IN_PARALLEL + " game types in parallel...\n");

        // Store results for each game type
        Map<String, GameResults> allResults = new ConcurrentHashMap<>();

        // Submit all games to run in parallel
        for (int i = 0; i < GAME_FILES.length; i++) {
            final int gameIndex = i;
            executor.submit(() -> {
                try {
                    String gameName = GAME_NAMES[gameIndex];
                    System.out.println("[" + gameName + "] Starting " + NUM_GAMES + " games...");
                    GameResults results = runGameMatch(GAME_FILES[gameIndex], gameName, false);
                    System.out.println("\n[" + gameName + "] Completed all " + NUM_GAMES + " games");
                    allResults.put(gameName, results);

                    // Print results for this game type
                    int totalGames = results.getWins(getAIName(PLAYER1_AI)) + results.getWins(getAIName(PLAYER2_AI)) + results.getDraws();
                    System.out.println("\n" + "=".repeat(60));
                    System.out.println("FINAL RESULTS FOR " + gameName.toUpperCase());
                    System.out.println("=" + " ".repeat(58) + "=\n");
                    printStatistics(results, totalGames);
                    System.out.println("\n" + "=".repeat(60) + "\n");

                    // Save results to file
                    saveResultsToFile(gameName, results, totalGames);

                } catch (Exception e) {
                    System.err.println("Error running " + GAME_NAMES[gameIndex] + ": " + e.getMessage());
                    e.printStackTrace();
                }
            });
        }

        // Shutdown the executor and wait for all tasks to complete
        executor.shutdown();
        try {
            if (!executor.awaitTermination(8, TimeUnit.HOURS)) {
                System.out.println("Forcing shutdown of remaining tasks...");
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            System.err.println("Interrupted while waiting for tasks to complete");
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Force exit to ensure all threads are terminated
        System.out.println("\nAll tasks completed. Exiting...");
        System.exit(0);

        // Print summary of all results
        System.out.println("\n" + "#".repeat(70));
        System.out.println("#" + " ".repeat(23) + "ALL GAMES COMPLETED" + " ".repeat(23) + "#");
        System.out.println("#".repeat(70) + "\n");

        for (String gameName : GAME_NAMES) {
            GameResults results = allResults.get(gameName);
            if (results != null) {
                int totalGames = results.getWins(getAIName(PLAYER1_AI)) + results.getWins(getAIName(PLAYER2_AI)) + results.getDraws();
                System.out.println("\n=== " + gameName + " ===");
                printStatistics(results, totalGames);
            }
        }

        System.out.println("\n" + "#".repeat(70));
        System.out.println("#" + " ".repeat(27) + "END OF REPORT" + " ".repeat(27) + "#");
        System.out.println("#".repeat(70) + "\n");
    }

    /**
     * Runs a match between two AIs for a specific game
     */
    private static boolean showDetailedOutput = true;

    private static class GameResults {
        private static final int BATCH_SIZE = 100;
        private final Map<String, AtomicInteger> results = new ConcurrentHashMap<>();
        private final AtomicInteger draws = new AtomicInteger(0);
        private final List<GameResult> currentBatch = new ArrayList<>();
        private final String player1Name;
        private final String player2Name;

        // Aggregated statistics
        private long totalGames = 0;
        private long totalSimulationsP1 = 0;
        private long totalSimulationsP2 = 0;
        private long totalTurnsP1 = 0;
        private long totalTurnsP2 = 0;

        public synchronized void addResult(GameResult result) {
            try {
                // Add to current batch
                currentBatch.add(result);

                // Update running totals for the batch
                if (result.wasDraw) {
                    draws.incrementAndGet();
                } else if (result.winner != null) {
                    results.computeIfAbsent(result.winner, k -> new AtomicInteger(0)).incrementAndGet();
                }

                // Process batch if we've reached batch size
                if (currentBatch.size() >= BATCH_SIZE) {
                    processBatch();
                }
            } catch (Exception e) {
                System.err.println("Error processing game result: " + e.getMessage());
                e.printStackTrace();
            }
        }

        private void processBatch() {
            if (currentBatch.isEmpty()) return;

            // Aggregate statistics for the current batch
            long batchSimsP1 = 0;
            long batchSimsP2 = 0;
            long batchTurnsP1 = 0;
            long batchTurnsP2 = 0;

            for (GameResult result : currentBatch) {
                batchSimsP1 += result.simulationsAI1;
                batchSimsP2 += result.simulationsAI2;
                batchTurnsP1 += result.turnsAI1;
                batchTurnsP2 += result.turnsAI2;
            }

            // Update running totals
            totalSimulationsP1 += batchSimsP1;
            totalSimulationsP2 += batchSimsP2;
            totalTurnsP1 += batchTurnsP1;
            totalTurnsP2 += batchTurnsP2;
            totalGames += currentBatch.size();

            // Clear the current batch
            currentBatch.clear();
        }

        public void finalizeStats() {
            // Process any remaining games in the last incomplete batch
            processBatch();
        }

        public double getAverageSimulations(String playerName) {
            if (totalGames == 0) return 0;
            long total = playerName.equals(player1Name) ? totalSimulationsP1 : totalSimulationsP2;
            return (double) total / totalGames;
        }

        public double getAverageSimulationsPerTurn(String playerName) {
            if (totalGames == 0) return 0;
            long totalSims = playerName.equals(player1Name) ? totalSimulationsP1 : totalSimulationsP2;
            long totalTurns = playerName.equals(player1Name) ? totalTurnsP1 : totalTurnsP2;
            return totalTurns > 0 ? (double) totalSims / totalTurns : 0;
        }

        public double getAverageTurns(String playerName) {
            if (totalGames == 0) return 0;
            long totalTurns = playerName.equals(player1Name) ? totalTurnsP1 : totalTurnsP2;
            return (double) totalTurns / totalGames;
        }

        // Inner class to hold game result data
        public static class GameResult {
            public final String gameName;
            public final int gameNumber;
            public final String winner;
            public final boolean wasDraw;
            public final long simulationsAI1;
            public final long simulationsAI2;
            public final int turnsAI1;
            public final int turnsAI2;

            public GameResult(String gameName, int gameNumber, String winner, boolean wasDraw,
                              long simulationsAI1, long simulationsAI2, int turnsAI1, int turnsAI2) {
                this.gameName = gameName;
                this.gameNumber = gameNumber;
                this.winner = winner;
                this.wasDraw = wasDraw;
                this.simulationsAI1 = simulationsAI1;
                this.simulationsAI2 = simulationsAI2;
                this.turnsAI1 = turnsAI1;
                this.turnsAI2 = turnsAI2;
            }
        }

        public GameResults(String player1Name, String player2Name) {
            this.player1Name = player1Name;
            this.player2Name = player2Name;
            results.put(player1Name, new AtomicInteger(0));
            results.put(player2Name, new AtomicInteger(0));
        }

        public void recordWin(String player) {
            results.get(player).incrementAndGet();
        }

        public void recordDraw() {
            draws.incrementAndGet();
        }

        public int getWins(String player) {
            return results.get(player).get();
        }

        public int getDraws() {
            return draws.get();
        }
    }

    private static class GameTask implements Callable<GameResults.GameResult> {
        private final String gameName;
        private final int gameNumber;
        private final boolean player1IsFirst;
        private final String player1Name;
        private final String player2Name;
        private final Game game;
        private final Trial trial;
        private final Context context;
        private final AI ai1;
        private final AI ai2;
        private long totalSimulationsAI1 = 0;
        private long totalSimulationsAI2 = 0;
        private int turnsAI1 = 0;
        private int turnsAI2 = 0;

        /**
         * Cleans up resources used by this game task
         */
        private void cleanup() {
            try {
                // Clean up AI resources
                if (ai1 != null) {
                    try {
                        ai1.closeAI();
                    } catch (Exception e) {
                        System.err.println("Error closing AI1: " + e.getMessage());
                    }
                }
                if (ai2 != null) {
                    try {
                        ai2.closeAI();
                    } catch (Exception e) {
                        System.err.println("Error closing AI2: " + e.getMessage());
                    }
                }

                // Clear any game state references
                if (context != null) {
                    // Reset the context to clear any game state
                    try {
                        trial.reset(game);
                        context.reset();
                    } catch (Exception e) {
                        System.err.println("Error resetting game context: " + e.getMessage());
                    }
                }

                // Clear any trial data by creating a new trial
                if (trial != null) {
                    try {
                        // Create a new empty trial to replace the old one
                        Trial newTrial = new Trial(game);
                        // Note: If the trial is final, we can't modify it, so we just let it be garbage collected
                    } catch (Exception e) {
                        System.err.println("Error creating new trial: " + e.getMessage());
                    }
                }

                // Suggest garbage collection
                System.gc();

            } catch (Exception e) {
                System.err.println("Error during cleanup: " + e.getMessage());
            } finally {
                // Ensure we clear any remaining references
                try {
                    if (ai1 != null) {
                        ai1.closeAI();
                    }
                    if (ai2 != null) {
                        ai2.closeAI();
                    }
                } catch (Exception e) {
                    // Ignore any errors during final cleanup
                    System.out.println(" // Ignore any errors during final cleanup" + e.getMessage());
                }
            }
        }

        GameTask(String gameName, int gameNumber, boolean player1IsFirst, String player1Name, String player2Name, Game game, Trial trial, Context context, AI ai1, AI ai2) {
            this.gameName = gameName;
            this.gameNumber = gameNumber;
            this.player1IsFirst = player1IsFirst;
            this.player1Name = player1Name;
            this.player2Name = player2Name;
            this.game = game;
            this.trial = trial;
            this.context = context;
            this.ai1 = ai1;
            this.ai2 = ai2;
        }

        @Override
        public GameResults.GameResult call() {
            String currentGameName = gameName + " (Game " + gameNumber + ")";
            if (showDetailedOutput) {
                System.out.println("\n[" + gameName + " - Game " + gameNumber + "] Starting - " +
                        (player1IsFirst ? player1Name : player2Name) + " moves first");
            }

            // Validate critical objects
            if (game == null || context == null || ai1 == null || ai2 == null) {
                System.err.println("[ERROR] Critical initialization error - missing required objects");
                return new GameResults.GameResult(gameName, gameNumber, null, true, 0, 0, 0, 0);
            }

            // Initialize game state with error handling
            try {
                if (game == null) {
                    throw new IllegalStateException("Game is null");
                }
                if (context == null) {
                    throw new IllegalStateException("Context is null");
                }

                // Initialize the game
                game.start(context);

                // Verify the game state after initialization
                if (context.state() == null || context.state().containerStates() == null) {
                    throw new IllegalStateException("Game state not properly initialized after game.start()");
                }

                // Initialize AIs
                try {
                    ai1.initAI(game, player1IsFirst ? 1 : 2);
                    ai2.initAI(game, player1IsFirst ? 2 : 1);
                } catch (Exception e) {
                    throw new IllegalStateException("Failed to initialize AIs: " + e.getMessage(), e);
                }

                if (showDetailedOutput) {
                    System.out.println("[" + gameName + " - Game " + gameNumber + "] Game initialized successfully");
                }

            } catch (Exception e) {
                String errorMsg = "[ERROR] Error initializing game or AIs: " + e.getMessage();
                System.err.println(errorMsg);
                if (e.getCause() != null) {
                    System.err.println("Caused by: " + e.getCause().getMessage());
                }
                return new GameResults.GameResult(gameName, gameNumber, null, true, 0, 0, 0, 0);
            }

            // Get model with null check
            final Model model = context.model();
            if (model == null) {
                System.err.println("[ERROR] Failed to get game model");
                return new GameResults.GameResult(gameName, gameNumber, null, true, 0, 0, 0, 0);
            }

            final int MAX_MOVES = 500;
            int moveCount = 0;
            long startTime = System.currentTimeMillis();

            while (moveCount < MAX_MOVES) {
                // Check for null trial or game over
                if (context.trial() == null || context.trial().over()) {
                    break;
                }

                List<AI> currentPlayerAI = new ArrayList<>();
                currentPlayerAI.add(null); // For player 0 (nonexistent)
                currentPlayerAI.add(ai1);  // For player 1
                currentPlayerAI.add(ai2);  // For player 2

                try {
                    long moveStart = System.currentTimeMillis();
                    //model.startNewStep(context, currentPlayerAI, TIME_PER_MOVE);
                    model.startNewStep(context, currentPlayerAI, TIME_PER_MOVE);
                    long moveTime = System.currentTimeMillis() - moveStart;

                    if (showDetailedOutput) {
                        System.out.println("Move " + (moveCount + 1) + " took " + moveTime + "ms");
                    }

                    // Track simulations and turns for each AI
                    int currentPlayer = context.state().playerToAgent(context.state().mover());
                    double simsThisTurn = 0;

                    if (currentPlayer == 1) {
                        if (ai1 instanceof PNSMCTS_L2_RAVE) {
                            simsThisTurn = ((PNSMCTS_L2_RAVE) ai1).getSimsThisTurn();
                        } else if (ai1 instanceof PNSMCTS_L2) {
                            simsThisTurn = ((PNSMCTS_L2) ai1).getSimsThisTurn();
                        } else if (ai1 instanceof PNSMCTS_L2_MAST) {
                            simsThisTurn = ((PNSMCTS_L2_MAST) ai1).getSimsThisTurn();
                        } else if (ai1 instanceof PNSMCTS_MAST) {
                            simsThisTurn = ((PNSMCTS_MAST) ai1).getSimsThisTurn();
                        }
                        totalSimulationsAI1 += simsThisTurn;
                        turnsAI1++;
                    } else if (currentPlayer == 2) {
                        if (ai2 instanceof PNSMCTS_L2_RAVE) {
                            simsThisTurn = ((PNSMCTS_L2_RAVE) ai2).getSimsThisTurn();
                        } else if (ai2 instanceof PNSMCTS_L2) {
                            simsThisTurn = ((PNSMCTS_L2) ai2).getSimsThisTurn();
                        } else if (ai2 instanceof PNSMCTS_L2_MAST) {
                            simsThisTurn = ((PNSMCTS_L2_MAST) ai2).getSimsThisTurn();
                        } else if (ai2 instanceof PNSMCTS_MAST) {
                            simsThisTurn = ((PNSMCTS_MAST) ai2).getSimsThisTurn();
                        }
                        totalSimulationsAI2 += simsThisTurn;
                        turnsAI2++;
                    }

                    // Metrics are now only shown at the end of the test

                    moveCount++;
                } catch (Exception e) {
                    System.err.println("[ERROR] Error during move " + (moveCount + 1) + ": " + e.getMessage());
                    break;
                }
            }

            long totalTime = System.currentTimeMillis() - startTime;
            if (showDetailedOutput) {
                System.out.println("[" + gameName + " - Game " + gameNumber + "] Completed in " +
                        moveCount + " moves (" + (totalTime/1000.0) + "s)");
            }

            try {
                // Record results
                String winnerName = null;
                boolean isDraw = false;

                // Check if trial is accessible
                if (context.trial() == null) {
                    System.err.println("[ERROR] Trial is null");
                    isDraw = true;
                }
                // Game ended by move limit
                else if (moveCount >= MAX_MOVES && !context.trial().over()) {
                    System.out.println("[FINISHED] " + currentGameName + " - Reached 500 move limit (Game " + gameNumber + ")");
                    isDraw = true;
                }
                // Game ended normally
                else if (context.trial().over()) {
                    try {
                        if (context.trial().status() == null) {
                            System.err.println("[ERROR] Game status is null");
                            isDraw = true;
                        } else {
                            int winner = context.trial().status().winner();
                            if (winner == 0) {
                                // No winner - count as draw
                                // Enhanced debug output
                                System.err.println("\n=== GAME END DEBUG ===");
//                                double[] scores = RankUtils.agentUtilities(context);
//                                System.out.println("Player 1's score: " + scores[0]);
//                                System.out.println("Player 2's score: " + scores[1]);
                                System.err.println("Game: " + currentGameName + " (Game " + gameNumber + ")");
                                System.err.println("Move count: " + moveCount);
                                System.err.println("Winner ID: " + winner);
                                System.err.println("Player 1: " + player1Name + (player1IsFirst ? " (First)" : " (Second)"));
                                System.err.println("Player 2: " + player2Name + (player1IsFirst ? " (Second)" : " (First)"));

                                // Log game state
                                try {
                                    System.err.println("Current player: " + context.state().playerToAgent(context.state().mover()));
                                    System.err.println("Available moves: " + game.moves(context).moves().size());

                                    // Log the last few moves if available
                                    int movesToShow = Math.min(5, context.trial().numMoves());
                                    if (movesToShow > 0) {
                                        System.err.println("Last " + movesToShow + " moves:");
                                        for (int i = Math.max(0, context.trial().numMoves() - movesToShow); i < context.trial().numMoves(); i++) {
                                            Move move = context.trial().getMove(i);
                                            System.err.println("  Move " + (i+1) + ": " + move + " by P" + move.mover());
                                        }
                                    }
                                } catch (Exception e) {
                                    System.err.println("Error getting game state: " + e.getMessage());
                                }

                                System.err.println("Game ended in a draw - No winner (winner=0)");
                                isDraw = true;
                            } else if ((player1IsFirst && winner == 1) || (!player1IsFirst && winner == 2)) {
                                System.out.println("Winner: " + player1Name + " " + currentGameName);
                                winnerName = player1Name;
                            } else {
                                System.out.println("Winner: " + player2Name + " " + currentGameName);
                                winnerName = player2Name;
                            }
                            //System.err.println("===================\n");
                        }
                    } catch (Exception e) {
                        System.err.println("[ERROR] Error determining winner: " + e.getMessage());
                        isDraw = true;
                    }
                }
                // Game ended unexpectedly
                else {
                    System.out.println("[FINISHED] " + currentGameName + " - Game ended unexpectedly (Game " + gameNumber + ")");
                    isDraw = true;
                }

                if (isDraw) {
                    if (showDetailedOutput) {
                        System.out.println("[FINISHED] " + currentGameName + " - Game ended in a draw");
                    }
                    return new GameResults.GameResult(gameName, gameNumber, null, true,
                            totalSimulationsAI1, totalSimulationsAI2, turnsAI1, turnsAI2);
                } else {
                    if (showDetailedOutput) {
                        System.out.println("[FINISHED] " + currentGameName + " - Winner: " + winnerName);
                    }
                    return new GameResults.GameResult(gameName, gameNumber, winnerName, false,
                            totalSimulationsAI1, totalSimulationsAI2, turnsAI1, turnsAI2);
                }

            } catch (Exception e) {
                System.err.println("[ERROR] Error in game result processing: " + e.getMessage());
                return new GameResults.GameResult(gameName, gameNumber, null, true, 0, 0, 0, 0);
            }
            finally {
                // Clean up resources once at the end
                cleanup();
            }
        }
    }

    private static class GameWorker implements Callable<GameResults.GameResult> {
        private final String gameFile;
        private final String gameName;
        private final int gameNumber;
        private final boolean player1IsFirst;

        public GameWorker(String gameFile, String gameName, int gameNumber, boolean player1IsFirst) {
            this.gameFile = gameFile;
            this.gameName = gameName;
            this.gameNumber = gameNumber;
            this.player1IsFirst = player1IsFirst;
        }

        @Override
        public GameResults.GameResult call() throws Exception {
            // Load a fresh game instance
            Game game = GameLoader.loadGameFromFile(new File(gameFile));
//            Game game = GameLoader.loadGameFromName(gameFile);
            Trial trial = new Trial(game);
            Context context = new Context(game, trial);
            AI ai1 = createAI(PLAYER1_AI, game, 1);
            AI ai2 = createAI(PLAYER2_AI, game, 2);

            try {
                // Create and run the game task
                GameTask task = new GameTask(
                    gameName, gameNumber, player1IsFirst,
                    getAIName(PLAYER1_AI), getAIName(PLAYER2_AI),
                    game, trial, context, ai1, ai2
                );
                
                return task.call();
            } finally {
                // Clean up resources
                if (ai1 instanceof AutoCloseable) {
                    try { ((AutoCloseable)ai1).close(); } catch (Exception e) {}
                }
                if (ai2 instanceof AutoCloseable) {
                    try { ((AutoCloseable)ai2).close(); } catch (Exception e) {}
                }
                if (ai1 instanceof PNSMCTS_L2_MAST) {
                    ((PNSMCTS_L2_MAST) ai1).cleanup();
                }
                if (ai2 instanceof PNSMCTS_L2_MAST) {
                    ((PNSMCTS_L2_MAST) ai2).cleanup();
                }
                // Finalize the game state
                context.trial().setStatus(null);
                context.reset();
            }
        }
    }

    private static GameResults runGameMatch(String gameFile, String gameName, boolean showAllOutput) {
        GameResults gameResults = new GameResults(getAIName(PLAYER1_AI), getAIName(PLAYER2_AI));
        showDetailedOutput = showAllOutput;

        if (showDetailedOutput) {
            System.out.println("\n" + "=".repeat(50));
            System.out.println("STARTING MATCH: " + gameName);
            System.out.println("=".repeat(50));
            System.out.println("\n=== AI Comparison ===");
            System.out.println("Game: " + gameName);
            System.out.println("Time per move: " + TIME_PER_MOVE + "s");
            System.out.println("Number of games: " + NUM_GAMES);
            System.out.println("Games in parallel: " + GAMES_IN_PARALLEL);
            System.out.println("Player 1: " + getAIName(PLAYER1_AI));
            System.out.println("Player 2: " + getAIName(PLAYER2_AI));
        } else {
            System.out.println("\nRunning " + gameName + " (" + GAMES_IN_PARALLEL + " parallel games at a time)...");
        }

        // Create a thread pool with the desired number of parallel games
        ExecutorService executor = Executors.newFixedThreadPool(GAMES_IN_PARALLEL);
        // Create a completion service to handle completed tasks
        CompletionService<GameResults.GameResult> completionService = new ExecutorCompletionService<>(executor);
        int completedGames = 0;
        int submittedGames = 0;

        // Submit initial batch of games
        for (int i = 0; i < Math.min(GAMES_IN_PARALLEL, NUM_GAMES); i++) {
            boolean player1First = (i % 2 == 0);
            completionService.submit(new GameWorker(gameFile, gameName, i + 1, player1First));
            submittedGames++;
            if (showDetailedOutput) {
                System.out.println("Started game " + (i + 1) + " (Player 1 " + (player1First ? "first" : "second") + ")");
            }
        }

        // Process completed games and submit new ones
        int nextGameNumber = GAMES_IN_PARALLEL + 1;
        while (completedGames < NUM_GAMES) {
            try {
                // Wait for any game to complete (with timeout to handle potential hangs)
                Future<GameResults.GameResult> completedFuture = completionService.poll(24, TimeUnit.HOURS);
                if (completedFuture == null) {
                    throw new TimeoutException("Timed out waiting for game completion");
                }
                
                // Process completed game
                GameResults.GameResult result = completedFuture.get();
                if (result != null) {
                    gameResults.addResult(result);
                }
                completedGames++;
                
                // Submit new game if there are more to run
                if (submittedGames < NUM_GAMES) {
                    boolean player1First = (nextGameNumber % 2 == 1);
                    completionService.submit(
                        new GameWorker(gameFile, gameName, nextGameNumber, player1First)
                    );
                    submittedGames++;
                    if (showDetailedOutput) {
                        System.out.println("Started game " + nextGameNumber + 
                                         " (Player 1 " + (player1First ? "first" : "second") + ")");
                    }
                    nextGameNumber++;
                }
                
                // Print progress
                if (showDetailedOutput) {
                    System.out.println("\n=== Progress Update (" + completedGames + "/" + NUM_GAMES + " games) ===");
                    printStatistics(gameResults, completedGames);
                    System.out.println("Currently running: " + (submittedGames - completedGames) + " games in parallel");
                } else {
                    System.out.print(".");
                    if (completedGames % 50 == 0) System.out.println();
                }
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("Game execution was interrupted: " + e.getMessage());
                break;
            } catch (ExecutionException e) {
                System.err.println("Error executing game task: " + e.getCause().getMessage());
                e.getCause().printStackTrace();
                // Continue with the next game
            } catch (Exception e) {
                System.err.println("An unexpected error occurred: " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        // Shutdown the executor
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Process any remaining completed futures to ensure all games complete
        while (completedGames < submittedGames) {
            try {
                Future<GameResults.GameResult> completedFuture = completionService.poll(5, TimeUnit.SECONDS);
                if (completedFuture == null) {
                    System.err.println("Timed out waiting for remaining games to complete");
                    break;
                }
                
                GameResults.GameResult result = completedFuture.get();
                if (result != null) {
                    gameResults.addResult(result);
                }
                completedGames++;

                // Print progress for the remaining games
                if (showDetailedOutput) {
                    System.out.println("\n=== Progress Update (" + completedGames + "/" + NUM_GAMES + " games) ===");
                    printStatistics(gameResults, completedGames);
                } else {
                    System.out.print(".");
                    if (completedGames % 50 == 0) System.out.println();
                }
            } catch (Exception e) {
                System.err.println("Error processing remaining game: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // Shutdown the executor and wait for all games to complete
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                System.err.println("Some games did not complete within the timeout period");
                executor.shutdownNow();
            } else if (completedGames == NUM_GAMES) {
                System.out.println("\n\n=== ALL " + NUM_GAMES + " GAMES COMPLETED SUCCESSFULLY ===");
                printStatistics(gameResults, completedGames);
            } else {
                System.out.println("\n\n=== COMPLETED " + completedGames + " OUT OF " + NUM_GAMES + " GAMES ===");
                printStatistics(gameResults, completedGames);
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        return gameResults;
    }

    private static void saveResultsToFile(String gameName, GameResults gameResults, int gamesPlayed) {
        // Create results directory if it doesn't exist
        Path resultsDir = Paths.get("results");
        try {
            if (!Files.exists(resultsDir)) {
                Files.createDirectories(resultsDir);
            }
        } catch (IOException e) {
            System.err.println("Error creating results directory: " + e.getMessage());
            return;
        }

        // Use a timestamp to create unique filenames
        String timestamp = new java.text.SimpleDateFormat("yyyyMMdd-HHmmss").format(new java.util.Date());
        String fileName = String.format("results/game_results_%s_%s.txt",
                gameName.replaceAll("\\s+", "_"), timestamp);

        try (PrintWriter pw = new PrintWriter(new FileWriter(fileName, true))) {
            // Write header
            pw.println("=".repeat(60));
            pw.println("FINAL RESULTS FOR " + gameName.toUpperCase());
            pw.println("=" + " ".repeat(58) + "=\n");

            // Calculate win rates
            int p1Wins = gameResults.getWins(getAIName(PLAYER1_AI));
            int p2Wins = gameResults.getWins(getAIName(PLAYER2_AI));
            int draws = gameResults.getDraws();

            // Write statistics
            pw.println("=== Game Statistics ===");
            pw.println(String.format("%-20s: %d", "Games completed", gamesPlayed));
            pw.println("-".repeat(40));

            // Format player names and results
            String p1Name = String.format("%-15s", getAIName(PLAYER1_AI));
            String p2Name = String.format("%-15s", getAIName(PLAYER2_AI));

            pw.println(String.format("%-20s: %d (%.1f%%)",
                    p1Name + " wins", p1Wins, (p1Wins * 100.0 / gamesPlayed)));
            pw.println(String.format("%-20s: %d (%.1f%%)",
                    p2Name + " wins", p2Wins, (p2Wins * 100.0 / gamesPlayed)));
            pw.println(String.format("%-20s: %d (%.1f%%)",
                    "Draws", draws, (draws * 100.0) / gamesPlayed));

            // Add average simulations and turns
            pw.println("\n=== Performance Metrics ===");
            // Get player names from gameResults
            String player1 = gameResults.results.keySet().stream().findFirst().orElse("Player1");
            String player2 = gameResults.results.keySet().stream().skip(1).findFirst().orElse("Player2");

            pw.println(String.format("%-30s: %,.1f", player1 + " avg sims/game", gameResults.getAverageSimulations(player1)));
            pw.println(String.format("%-30s: %,.1f", player2 + " avg sims/game", gameResults.getAverageSimulations(player2)));
            pw.println(String.format("%-30s: %,.1f", player1 + " avg sims/turn", gameResults.getAverageSimulationsPerTurn(player1)));
            pw.println(String.format("%-30s: %,.1f", player2 + " avg sims/turn", gameResults.getAverageSimulationsPerTurn(player2)));
            pw.println(String.format("%-30s: %,.1f", player1 + " avg turns/game", gameResults.getAverageTurns(player1)));
            pw.println(String.format("%-30s: %,.1f", player2 + " avg turns/game", gameResults.getAverageTurns(player2)));
            pw.println("-".repeat(40));

            // Write footer
            pw.println("-".repeat(40));
            pw.println("\nResults saved to: " + new File(fileName).getAbsolutePath());
            pw.println("=".repeat(60) + "\n");

            System.out.println("Results successfully saved to: " + new File(fileName).getAbsolutePath());
        } catch (IOException e) {
            System.err.println("Error writing to results file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void printStatistics(GameResults gameResults, int gamesPlayed) {
        System.out.println("\n=== Current Win Rates ===");
        System.out.println("Games completed: " + gamesPlayed);
        System.out.println("-".repeat(40));

        // Calculate and sort by win rate
        List<Map.Entry<String, AtomicInteger>> sortedEntries = new ArrayList<>(gameResults.results.entrySet());
        sortedEntries.sort((a, b) -> b.getValue().get() - a.getValue().get());

        for (Map.Entry<String, AtomicInteger> entry : sortedEntries) {
            int wins = entry.getValue().get();
            double winRate = (wins * 100.0) / gamesPlayed;
            System.out.printf("%-20s: %3d wins (%.1f%%)\n",
                    entry.getKey(),
                    wins,
                    winRate);
        }

        int draws = gameResults.getDraws();
        System.out.printf("%-20s: %3d (%.1f%%)\n",
                "Draws",
                draws,
                (draws * 100.0) / gamesPlayed);

        // Add average simulations and turns per AI
        String player1Name = getAIName(PLAYER1_AI);
        String player2Name = getAIName(PLAYER2_AI);

        System.out.println("\n=== Performance Metrics ===");
        System.out.printf("%-30s: %,.1f\n", player1Name + " avg sims/game", gameResults.getAverageSimulations(player1Name));
        System.out.printf("%-30s: %,.1f\n", player2Name + " avg sims/game", gameResults.getAverageSimulations(player2Name));
        System.out.printf("%-30s: %,.1f\n", player1Name + " avg sims/turn", gameResults.getAverageSimulationsPerTurn(player1Name));
        System.out.printf("%-30s: %,.1f\n", player2Name + " avg sims/turn", gameResults.getAverageSimulationsPerTurn(player2Name));
        System.out.printf("%-30s: %,.1f\n", player1Name + " avg turns/game", gameResults.getAverageTurns(player1Name));
        System.out.printf("%-30s: %,.1f\n", player2Name + " avg turns/game", gameResults.getAverageTurns(player2Name));
        System.out.println("-".repeat(40));
    }
}
