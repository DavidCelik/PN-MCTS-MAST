package mcts;

import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.RankUtils;
import other.context.Context;
import other.move.Move;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class PNSMCTS_MAST extends AI {


    public static boolean FIN_MOVE_SEL = false;
    public static int SOLVERLIKE_MINVISITS = Integer.MAX_VALUE; // 5; // Integer.MAX_VALUE;


    //-------------------------------------------------------------------------

    /**
     * Our player index
     */
    protected int player = -1;

    /**
     * Settings contain (in order): pnConstant, explorationConstant, time per turn
     */
    private static double[] settings; // PN-Constant, MCTS-Constant, Time per turn


    // Used to count simulations per second
    private static double sims = 0;
    private static double simsThisTurn = 0;
    private static double turns = 0;

    //-------------------------------------------------------------------------

    //-----------MAST---------------------------
    /**
     * @return The number of simulations performed in the current turn
     */
    public double getSimsThisTurn() {
        return simsThisTurn;
    }
    //-----------MAST---------------------------

    private final Map<PNSMCTS_MAST.NGramKey, Integer> nGramLastSeen = new HashMap<>();
    private final Map<PNSMCTS_MAST.NGramKey, Integer> opponentNGramLastSeen = new HashMap<>();
    private int totalSimulations = 0;
    private int nGramAdditions = 0;

    // N-gram NST related fields
    private static class NGramKey {
        final List<Move> sequence;

        NGramKey(List<Move> sequence) {
            this.sequence = new ArrayList<>(sequence);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof PNSMCTS_MAST.NGramKey)) return false;
            PNSMCTS_MAST.NGramKey nGramKey = (PNSMCTS_MAST.NGramKey) o;
            return sequence.equals(nGramKey.sequence);
        }

        @Override
        public int hashCode() {
            return sequence.hashCode();
        }
    }


    private final Map<Move, Map<PNSMCTS_MAST.NGramKey, Double>> nGramScores = new HashMap<>();
    private final Map<Move, Map<PNSMCTS_MAST.NGramKey, Integer>> nGramVisits = new HashMap<>();

    // N-gram statistics for the opponent
    private final Map<Move, Map<PNSMCTS_MAST.NGramKey, Double>> opponentNGramScores = new HashMap<>();
    private final Map<Move, Map<PNSMCTS_MAST.NGramKey, Integer>> opponentNGramVisits = new HashMap<>();

    // Move scores for both players (1-gram scores)
    private final Map<Move, Double> aiMoveScores = new HashMap<>();
    private final Map<Move, Integer> aiMoveVisits = new HashMap<>();
    private final Map<Move, Double> opponentMoveScores = new HashMap<>();
    private final Map<Move, Integer> opponentMoveVisits = new HashMap<>();

    private final Object statsLock = new Object();

    // Class to store a move along with the player who made it
    private static class MoveWithPlayer {
        final Move move;
        final int player;

        MoveWithPlayer(Move move, int player) {
            this.move = move;
            this.player = player;
        }

        @Override
        public String toString() {
            return String.format("P%d:%s", player, move);
        }
    }

    // Move history for the current simulation
    private final List<PNSMCTS_MAST.MoveWithPlayer> simulationMoveHistory = new ArrayList<>();
    private static final Random random = new Random();
    private static final boolean printDebug = false;  // Set to true to see UCB debug output
    private static final boolean doDecay = false;
    private final int nGramSize;
    private double simulationDecayFactor = 1.0; //0.94 Gamma value for decay (1.0 = no decay, 0.0 = only most recent matters) // N in N-gram (1 = MAST, 2 = bigram, 3 = trigram, etc.)
    private double moveDecayFactor = 1.0; //0.94 Gamma value for decay (1.0 = no decay, 0.0 = only most recent matters) // N in N-gram (1 = MAST, 2 = bigram, 3 = trigram, etc.)

    // Epsilon-greedy exploration parameter (20% exploration rate as per the paper)
    private static final double EPSILON = 0.2; // Reduced exploration rate for more exploitation in Connect 4
    private static final double newMoveExploration = 100.0;
    private static final int MIN_VISITS = 7; //k = 7
    // Decay interval for n-gram statistics (1 = decay every update)
    private static final int DECAY_INTERVAL = 1;

    // Option to make opponent play randomly
    private boolean opponentPlaysRandomly = false;

//-------------------------------------------------------------------------
    /**
     * Constructor
     */
    public PNSMCTS_MAST() {
        this.nGramSize = 1; // Default to 1-gram (MAST)
        this.friendlyName = "PNS_MAST UCT";
        double[] defaultSettings = {1.0, Math.sqrt(2), 1.0}; // PN-Constant, MCTS-Constant, Time per turn
        this.settings = defaultSettings;
    }

    public PNSMCTS_MAST(boolean finMove, int minVisits, double pnCons, int nGramSize) {
        this.nGramSize = Math.max(1, Math.min(3, nGramSize));
        this.FIN_MOVE_SEL = finMove;
        this.SOLVERLIKE_MINVISITS = minVisits;
        this.friendlyName = "PNS_MAST UCT";
        double[] defaultSettings = {pnCons, Math.sqrt(2), 1.0}; // PN-Constant, MCTS-Constant, Time per turn
        this.settings = defaultSettings;
    }

    public PNSMCTS_MAST(double[] settings) {
        this.nGramSize = 1; // Default to 1-gram (MAST)
        this.friendlyName = "PNS_MAST UCT";
        this.settings = settings;
    }


    //-------------------------------------------------------------------------

    @Override
    public Move selectAction(
            final Game game,
            final Context context,
            final double maxSeconds,
            final int maxIterations,
            final int maxDepth
    ) {
//        this.simsThisTurn = this.sims;
//        this.turns++;
        this.simsThisTurn = 0;
        this.turns++;

        // Start out by creating a new root node (no tree reuse in this example)
        final Node root = new Node(null, null, context, player);

        // We'll respect any limitations on max seconds and max iterations (don't care about max depth)
        final long stopTime = (maxSeconds > 0.0) ? System.currentTimeMillis() + (long) (maxSeconds * 1000L) : Long.MAX_VALUE;
        final int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;

        int numIterations = 0;

        // Our main loop through MCTS iterations
        while (
                numIterations < maxIts &&                    // Respect iteration limit
                        System.currentTimeMillis() < stopTime &&    // Respect time limit
                        !wantsInterrupt                                // Respect GUI user clicking the pause button
        ) {
            // Start in root node
            Node current = root;

            // Traverse tree
            while (true) {
                if (current.context.trial().over()) {
                    // We've reached a terminal state
                    break;
                }

                current = select(current);

                if (current.visitCount == 0) {
                    // We've expanded a new node, time for playout!
                    break;
                }
            }

            Context contextEnd = current.context;

            int numMoves = 0;
            if (!contextEnd.trial().over()) {
                // MAST-guided playout implementation
                contextEnd = new Context(contextEnd);


//                System.out.println("Starting playout simulation. AI is player: " + this.player);
//                System.out.println("Initial mover: " + contextEnd.state().mover());


                // Perform MAST-guided playout with (random) moves for opponent
                while (!contextEnd.trial().over()) {  // Continue until game ends
                    int currentPlayer = contextEnd.state().mover();


                    // Get legal moves for current player
                    FastArrayList<Move> legalMoves = game.moves(contextEnd).moves();
                    if (legalMoves.isEmpty()) {
                        break;  // Or handle no legal moves
                    }
                    // Log current player and available moves
//                    System.out.printf("\nMove %d - Player %d's turn%n", numMoves, currentPlayer);
//                    System.out.println("Legal moves: " + legalMoves.size());
                    // Select move based on player type
                    Move move;
                    if (currentPlayer != this.player && opponentPlaysRandomly) {
                        // Select random move for opponent if enabled
                        move = legalMoves.get(contextEnd.rng().nextInt(legalMoves.size()));
                    } else {
                        // Use N-gram statistics for move selection
                        move = selectMoveByMAST(legalMoves, currentPlayer);
                    }

//                    System.out.println("Selected move: " + move.getActionsWithConsequences(contextEnd));
                    // Make a deep copy of the context before applying the move
                    //Context contextBeforeMove = new Context(contextEnd);

                    // Apply the selected move
                    game.apply(contextEnd, move);
                    simulationMoveHistory.add(new PNSMCTS_MAST.MoveWithPlayer(move, currentPlayer));
                    // Add the move to the simulation history for N-gram statistics

                    numMoves++;
//                    System.out.println("After move - Next player: " + contextEnd.state().mover());
//                    System.out.println("Game over: " + contextEnd.trial().over());
                }
                // Update N-gram statistics based on the final game outcome
                if (!simulationMoveHistory.isEmpty()) {
                    // Get utilities for all players using Ludii's standard method
                    double[] utilities = RankUtils.utilities(contextEnd);
                    //System.out.println("utilities for updateNGramStats(utilities): " + Arrays.toString(utilities));
                    updateNGramStats(utilities);
                    simulationMoveHistory.clear();
                }
//                System.out.println("\n=== Playout Complete ===");
//                System.out.println("Total moves: " + (numMoves));
//                System.out.println("Final utilities: " + Arrays.toString(RankUtils.utilities(contextEnd)));
                sims++;
                simsThisTurn++;
            }

            // This computes utilities for all players at the of the playout,
            // which will all be values in [-1.0, 1.0]
            final double[] utilities = RankUtils.utilities(contextEnd);

            // Backpropagate utilities through the tree
            boolean changed = true;
            boolean firstNode = true;
            while (current != null) {
                current.visitCount += 1;
                for (int p = 1; p <= game.players().count(); ++p) {
                    current.scoreSums[p] += utilities[p];
                }
                if (!firstNode) {
                    if (changed) {
                        changed = current.setProofAndDisproofNumbers();
                        if (current.getChildren().size() > 0) {
                            current.setChildRanks();
                        }
                    }
                } else {
                    firstNode = false;
                }

                current = current.parent;
            }
            // if proofNum of root changed -> check is proven or disproven
            // if (changed) {
            //     for (Node child : root.children) {
            // if root is proven -> stop searching
            // if (child.proofNum == 0) { // causes problems with robust child final move selection
            //     return finalMoveSelection(root);
            // }
            // }
            // }

            // Increment iteration count
            ++numIterations;
        }

        // Apply decay if needed
        if (doDecay ) {
            applyMoveDecay(nGramScores, nGramVisits);
            applyMoveDecay(opponentNGramScores, opponentNGramVisits);
        }

        // Return the move we wish to play
        return finalMoveSelection(root);
    }

    /**
     * Selects child of the given "current" node according to UCT-PN equation.
     * This method also implements the "Expansion" phase of MCTS, and creates
     * new nodes if the given current node has unexpanded moves.
     *
     * @param current
     * @return Selected node (if it has 0 visits, it will be a newly-expanded node).
     */
    public static Node select(final Node current) {
        // All child nodes are created and added to the child list of the current node
        if (!current.expanded) {
            return current.developNode();
        }

        // Don't use UCT-PN until all nodes have been visited once
        if (current.getUnexpandedChildren().size() > 0) {
            return current.getUnexpandedChildren().remove(ThreadLocalRandom.current().nextInt(current.unexpandedChildren.size()));
        }

        // use UCT-PN equation to select from all children, with random tie-breaking
        Node bestChild = null; //current.children.get(0); //null;
        double bestValue = Double.NEGATIVE_INFINITY;
        int numBestFound = 0;

        double explorationConstant = settings[1];

        double pnConstant = settings[0];
        double total = current.getChildren().size();

        final int numChildren = current.children.size();
        final int mover = current.context.state().mover();

        for (int i = 0; i < numChildren; ++i) {
            final Node child = current.children.get(i);

            if (current.proofNum != 0 && current.disproofNum != 0) {
                if (child.proofNum == 0 && child.visitCount > SOLVERLIKE_MINVISITS) continue;
                if (child.disproofNum == 0 && child.visitCount > SOLVERLIKE_MINVISITS) continue;
            }

            final double exploit = child.scoreSums[mover] / child.visitCount;
            final double explore = Math.sqrt((Math.log(current.visitCount)) / child.visitCount); //UCT with changeable exploration constant
            final double pnEffect = 1 - (child.getRank() / total); // This formula assures that the node with lowest rank (best node) has the highest pnEffect

            // UCT-PN Formula
            final double uctValue = exploit + (explorationConstant * explore) + (pnConstant * pnEffect);

            if (uctValue > bestValue) {
                bestValue = uctValue;
                bestChild = child;
                numBestFound = 1;
            } else if (uctValue == bestValue && ThreadLocalRandom.current().nextInt() % ++numBestFound == 0) {
                // this case implements random tie-breaking
                bestChild = child;
            }
        }

        //if (bestChild==null) System.out.println("YYYY");
        return bestChild;
    }

    /**
     * Selects the move we wish to play using the "Robust Child" strategy
     * (meaning that we play the move leading to the child of the root node
     * with the highest visit count).
     *
     * @param rootNode
     * @return Final move as selected by PN-MCTS
     */
    public static Move finalMoveSelection(final Node rootNode) {
        Node bestChild = null;
        int bestVisitCount = Integer.MIN_VALUE;
        int numBestFound = 0;

        final int numChildren = rootNode.children.size();

        for (int i = 0; i < numChildren; ++i) {
            final Node child = rootNode.children.get(i);
            final int visitCount = child.visitCount;

            if (visitCount > bestVisitCount) {
                bestVisitCount = visitCount;
                bestChild = child;
                numBestFound = 1;
            } else if (visitCount == bestVisitCount && ThreadLocalRandom.current().nextInt() % ++numBestFound == 0) {
                // this case implements random tie-breaking
                bestChild = child;
            }
        }

        // To ensure a proven node will select the proven child too
        if (FIN_MOVE_SEL) {
            //System.out.println("XXXXXXX");
            if (rootNode.proofNum == 0) {
                for (Node child : rootNode.children) {
                    if (child.proofNum == 0) {
                        bestChild = child;
                        break;
                    }
                }
            }
        }

        return bestChild.moveFromParent;
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
    }

    @Override
    // Notifies Ludii if the game is playable by PN-MCTS
    public boolean supportsGame(final Game game) {
        if (game.isStochasticGame())
            return false;

        if (!game.isAlternatingMoveGame())
            return false;

        return true;
    }

    //-------------------------------------------------------------------------

    /**
     * Inner class for nodes used by example UCT
     *
     * @author Dennis Soemers
     */
    private static class Node implements Comparable<Node> {

        /**
         * Our parent node
         */
        private final Node parent;

        /**
         * The move that led from parent to this node
         */
        private final Move moveFromParent;

        /**
         * This objects contains the game state for this node (this is why we don't support stochastic games)
         */
        private final Context context;

        /**
         * Visit count for this node
         */
        private int visitCount = 0;

        /**
         * For every player, sum of utilities / scores backpropagated through this node
         */
        private final double[] scoreSums;

        /**
         * Child nodes
         */
        private final List<Node> children = new ArrayList<Node>();

        /**
         * List of moves for which we did not yet create a child node
         */
        private final FastArrayList<Move> unexpandedMoves;

        private final List<Node> unexpandedChildren = new ArrayList<Node>();

        /**
         * Flag to keep track of if a node has expanded its children yet
         */
        private boolean expanded = false;

        /**
         * Proof and Disproof number of current node
         */
        private double proofNum;
        private double disproofNum;

        /**
         * Rank of a node compared to "siblings". Needed for UCT-PN. Ranks ordered best to worst
         */
        private int rank;

        /**
         * Various necessary information variables.
         */
        private final PNSNodeTypes type;

        private PNSNodeValues value;

        public final int proofPlayer;

        public enum PNSNodeTypes {
            /**
             * An OR node
             */
            OR_NODE,

            /**
             * An AND node
             */
            AND_NODE
        }

        /**
         * Values of nodes in search trees in PNS
         */
        public enum PNSNodeValues {
            /**
             * A proven node
             */
            TRUE,

            /**
             * A disproven node
             */
            FALSE,

            /**
             * Unknown node (yet to prove or disprove)
             */
            UNKNOWN
        }


        /**
         * Constructor
         *
         * @param parent
         * @param moveFromParent
         * @param context
         */
        public Node(final Node parent, final Move moveFromParent, final Context context, final int proofPlayer) {
            this.parent = parent;
            this.moveFromParent = moveFromParent;
            this.context = context;
            final Game game = context.game();
            this.proofPlayer = proofPlayer;
            scoreSums = new double[game.players().count() + 1];
            // Set node type
            if (context.state().mover() == proofPlayer) {
                this.type = PNSNodeTypes.OR_NODE;
            } else {
                this.type = PNSNodeTypes.AND_NODE;
            }
            evaluate();
            setProofAndDisproofNumbers();

            // For simplicity, we just take ALL legal moves.
            // This means we do not support simultaneous-move games.
            unexpandedMoves = new FastArrayList<Move>(game.moves(context).moves());

            if (parent != null)
                parent.children.add(this);
        }

        /**
         * Evaluates a node as in PNS according to L. V. Allis' "Searching for Solutions in Games and Artificial Intelligence"
         */
        public void evaluate() {
            if (this.context.trial().over()) {
                if (RankUtils.utilities(this.context)[proofPlayer] == 1.0) {
                    this.value = PNSNodeValues.TRUE;
                } else {
                    this.value = PNSNodeValues.FALSE;
                }
            } else {
                this.value = PNSNodeValues.UNKNOWN;
            }
        }


        /**
         * Sets the proof and disproof values of the current node as it is done for PNS in L. V. Allis' "Searching for
         * Solutions in Games and Artificial Intelligence". Set differently depending on if the node has children yet.
         *
         * @return Returns true if something was changed and false if not. Used to improve PN-MCTS speed
         */
        public boolean setProofAndDisproofNumbers() {
            // If this node has child nodes
            if (this.expanded) {
                if (this.type == PNSNodeTypes.AND_NODE) {
                    double proof = 0;
                    for (int i = 0; i < this.children.size(); i++) {
                        proof += this.children.get(i).getProofNum();
                    }
                    double disproof = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < this.children.size(); i++) {
                        if (this.children.get(i).getDisproofNum() < disproof) {
                            disproof = this.children.get(i).getDisproofNum();
                        }
                    }
                    //If nothing changed return false
                    if (this.proofNum == proof && this.disproofNum == disproof) {
                        return false;
                    } else {
                        this.proofNum = proof;
                        this.disproofNum = disproof;
                        return true;
                    }
                } else if (this.type == PNSNodeTypes.OR_NODE) {
                    double disproof = 0;
                    for (int i = 0; i < this.children.size(); i++) {
                        disproof += this.children.get(i).getDisproofNum();
                    }
                    double proof = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < this.children.size(); i++) {
                        if (this.children.get(i).getProofNum() < proof) {
                            proof = this.children.get(i).getProofNum();
                        }
                    }
                    //If nothing changed return false
                    if (this.proofNum == proof && this.disproofNum == disproof) {
                        return false;
                    } else {
                        this.proofNum = proof;
                        this.disproofNum = disproof;
                        return true;
                    }
                }
            } else if (!this.expanded) {
                // (Dis)proof numbers are set according to evaluation until properly checked
                if (this.value == PNSNodeValues.FALSE) {
                    this.proofNum = Double.POSITIVE_INFINITY;
                    this.disproofNum = 0;
                } else if (this.value == PNSNodeValues.TRUE) {
                    this.proofNum = 0;
                    this.disproofNum = Double.POSITIVE_INFINITY;
                } else if (this.value == PNSNodeValues.UNKNOWN) {
                    this.proofNum = 1;
                    this.disproofNum = 1;
                }
            }
            //If we haven't expanded yet it will definitely be changed so return true
            return true;
        }

        /**
         * Develops a node by adding all the children nodes. Then returns one child at random for the selection phase.
         *
         * @return One of the new child nodes
         */
        public Node developNode() {
            if (this.value == PNSNodeValues.UNKNOWN) {
                for (int i = 0; i < this.unexpandedMoves.size(); i++) {
                    final Move move = this.unexpandedMoves.get(i);
                    final Context context = new Context(this.context);
                    context.game().apply(context, move);
                    Node node = new Node(this, move, context, this.proofPlayer);
                    unexpandedChildren.add(node);
                }
                this.expanded = true;
                //this.setProofAndDisproofNumbers();
                return this.unexpandedChildren.remove(ThreadLocalRandom.current().nextInt(this.unexpandedChildren.size()));
            } else {
                this.expanded = true;
                return this;
            }
        }

        /**
         * Set an ordered ranking for the UCT-PN formula in the selection step of MCTS
         */

        public void setChildRanks() {
            List<Node> sorted = new ArrayList<Node>(this.children);
            Collections.sort(sorted);
            Node lastNode = null;
            for (int i = 0; i < sorted.size(); i++) {
                Node child = sorted.get(i);
                // If there's a tie
                if (lastNode != null && this.type == PNSNodeTypes.OR_NODE && lastNode.getProofNum() == child.getProofNum()) {
                    child.setRank(lastNode.getRank());
                    // If there's a tie
                } else if (lastNode != null && this.type == PNSNodeTypes.AND_NODE && lastNode.getDisproofNum() == child.getDisproofNum()) {
                    child.setRank(lastNode.getRank());
                } else {
                    child.setRank(i + 1);
                }
                lastNode = child;
            }
        }

        public List<Node> getChildren() {
            return children;
        }

        public double getProofNum() {
            return proofNum;
        }

        public double getDisproofNum() {
            return disproofNum;
        }

        public PNSNodeTypes getType() {
            return type;
        }

        public int getRank() {
            return rank;
        }

        public List<Node> getUnexpandedChildren() {
            return unexpandedChildren;
        }

        public void setRank(int rank) {
            this.rank = rank;
        }

        // Used to rank children
        @Override
        public int compareTo(Node o) {
            if (this.parent.getType() == PNSNodeTypes.OR_NODE) {
                if (this.getProofNum() < o.getProofNum()) {
                    return -1;
                } else if (this.getProofNum() > o.getProofNum()) {
                    return 1;
                } else {
                    return 0;
                }
            } else if (this.parent.getType() == PNSNodeTypes.AND_NODE) {
                if (this.getDisproofNum() < o.getDisproofNum()) {
                    return -1;
                } else if (this.getDisproofNum() > o.getDisproofNum()) {
                    return 1;
                } else {
                    return 0;
                }
            }
            return 0;
        }
    }

    //-------------------------------------------------------------------------

    //-------------------------------------------------------------------------

    /**
     * Gets the current N-gram size
     *
     * @return The current N-gram size (1 for MAST, 2 for bigram, etc.)
     */

    public int getNGramSize() {
        return nGramSize;
    }

    /**
     * Prints all N-gram statistics to the console
     *
     * @param maxPerSize Maximum number of N-grams to print per size (1 to nGramSize)
     */
    public void printNGramStats(int maxPerSize, Context context) {
        System.out.println("\n=== N-gram Statistics (N = " + nGramSize + ") ===");

        // Group N-grams by their size
        Map<Integer, List<Map.Entry<PNSMCTS_MAST.NGramKey, Integer>>> ngramsBySize = new HashMap<>();

        // First, collect all n-grams from nGramVisits
        for (Map<PNSMCTS_MAST.NGramKey, Integer> ngramMap : nGramVisits.values()) {
            for (Map.Entry<PNSMCTS_MAST.NGramKey, Integer> entry : ngramMap.entrySet()) {
                PNSMCTS_MAST.NGramKey ngram = entry.getKey();
                int size = ngram.sequence.size();
                ngramsBySize.computeIfAbsent(size, k -> new ArrayList<>()).add(entry);
            }
        }

        // Print N-grams by size
        for (int size = 1; size <= nGramSize; size++) {
            List<Map.Entry<PNSMCTS_MAST.NGramKey, Integer>> ngrams = ngramsBySize.getOrDefault(size, Collections.emptyList());
            System.out.printf("\n=== %d-grams (showing %d/%d) ===\n",
                    size, Math.min(ngrams.size(), maxPerSize), ngrams.size());

            // Print N-grams in their natural order (no sorting)
            int count = 0;
            for (Map.Entry<PNSMCTS_MAST.NGramKey, Integer> entry : ngrams) {
                if (count++ >= maxPerSize) break;
                PNSMCTS_MAST.NGramKey ngram = entry.getKey();
                int visits = entry.getValue();

                // Find the score for this n-gram (need to search through all moves)
                Double score = null;
                for (Map<PNSMCTS_MAST.NGramKey, Double> scoreMap : nGramScores.values()) {
                    if (scoreMap.containsKey(ngram)) {
                        score = scoreMap.get(ngram);
                        break;
                    }
                }

                // Skip if we don't have a score for this N-gram
                if (score == null) {
                    continue;
                }

                double avgScore = score / visits;

                // Handle null context
                String moveInfo = (context != null) ?
                        moveSequenceWithPlayerInfo(ngram.sequence, context) :
                        moveSequenceToString(ngram.sequence.stream()
                                .map(move -> new PNSMCTS_MAST.MoveWithPlayer(move, -1)) // Default player ID since we don't have context
                                .collect(Collectors.toList()));

                System.out.printf("  %s - Visits: %d, Avg Score: %.4f\n",
                        moveInfo, visits, avgScore);
            }
        }
    }

    /**
     * Converts a sequence of moves to a readable string with player information
     */
    /**
     * Converts a sequence of moves to a detailed string with player information
     * including the current player for each move
     */
    private String moveSequenceWithPlayerInfo(List<Move> moves, Context context) {
        if (moves.isEmpty()) return "[]";

        // Create a copy of the context to simulate moves
        Context tempContext = new Context(context);
        StringBuilder sb = new StringBuilder();

        sb.append("\n");
        for (int i = 0; i < moves.size(); i++) {
            Move move = moves.get(i);
            int mover = move.mover();
            int currentPlayerBeforeMove = tempContext.state().mover();

            sb.append(String.format("Move %d: P%d (current: P%d) - %s%n",
                    i + 1,
                    mover,
                    currentPlayerBeforeMove,
                    move.toString()));

            // Apply the move to update the context for the next move
            if (tempContext.active()) {  // Only apply if the game is still active
                tempContext.game().apply(tempContext, move);
                // The game state is already updated by game().apply()
            }
        }
        return sb.toString();
    }

    /**
     * Simple move sequence to string conversion
     */
    private String moveSequenceToString(List<PNSMCTS_MAST.MoveWithPlayer> moves) {
        if (moves == null || moves.isEmpty()) return "";
        return moves.stream()
                .map(mwp -> mwp.move.toString()) // Just use the move's string representation
                .collect(Collectors.joining(" -> "));
    }


    /**
     * Selects a move using N-gram statistics with epsilon-greedy exploration.
     * Uses the getNGramScore method to evaluate moves, which considers:
     * - Minimum visits threshold for n-grams
     * - Weighted averages of n-gram scores
     * - Fallback to 1-gram scores when needed
     * - High exploration value for completely new moves
     *
     * @param legalMoves List of legal moves to choose from
     * @return The best move according to n-gram statistics, or a random move if no data available
     */
    private Move selectMoveByMAST(FastArrayList<Move> legalMoves, int currentPlayer) {
        if (legalMoves == null || legalMoves.isEmpty()) {
            return null;
        }

        // With probability EPSILON, select a random move (exploration)
        if (random.nextDouble() < EPSILON) {
            Move randomMove = legalMoves.get(random.nextInt(legalMoves.size()));
            if (printDebug) {
                System.out.println("\n--- Random Move Selection (Exploration) ---");
                System.out.println("Selected random move: " + randomMove);
            }
            return randomMove;
        }

        // Track the best move based on move scores
        Move bestMove = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        if (printDebug) {
            System.out.println("\n--- Move Selection (Scores) ---");
            System.out.println("Move\tScore");
            System.out.println("----------------------");
        }

        // Find the move with the highest score
        for (Move move : legalMoves) {
            double score = getMoveScore(move, currentPlayer);

            if (printDebug) {
                System.out.printf("%s\t%.4f%n",
                        move.toString().replaceAll("\n", " "),
                        score);
            }

            // Track the move with the highest score
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        // If no move was selected (shouldn't happen, but just in case)
        if (bestMove == null) {
            return legalMoves.get(random.nextInt(legalMoves.size()));
        }

        return bestMove;
    }

    /**
     * Gets the N-gram score for a move, considering both AI and opponent statistics
     *
     * @param move          The move to evaluate
     * @param currentPlayer The player making the move
     * @return The weighted average score for the move based on N-gram statistics
     */

    private double getMoveScore(Move move, int currentPlayer) {
        // For 1-grams, use the precomputed move scores
        if (currentPlayer == this.player) {
            return aiMoveScores.getOrDefault(move, newMoveExploration);
        } else {
            return opponentMoveScores.getOrDefault(move, newMoveExploration);
        }
    }




    private void updateNGramStats(double[] utilities) {
        if (simulationMoveHistory.isEmpty()) return;

//        System.out.println("\n=== N-gram Update ===");
//        System.out.println("Move history size: " + simulationMoveHistory.size());
//        System.out.println("Utilities: " + Arrays.toString(utilities));
//
//        // Print the full simulation history first
//        System.out.println("\nSimulation Move History:");
//        for (int i = 0; i < simulationMoveHistory.size(); i++) {
//            MoveWithPlayer mwp = simulationMoveHistory.get(i);
//            System.out.printf("  [%d] P%d: %s%n",
//                    i + 1,
//                    mwp.player,
//                    mwp.move.getActionsWithConsequences(contextEnd));
//        }

        // Optimize: Use local variable to avoid repeated volatile reads
        int currentTotalSimulations = ++totalSimulations;

        // Pre-size collections based on typical usage
        Map<Integer, Map<PNSMCTS_MAST.NGramKey, Integer>> aggregatedPathNGramCounts = new HashMap<>(2);

        // Reusable list for n-gram sequences to avoid object creation in inner loops
        List<Move> sequenceBuilder = new ArrayList<>(nGramSize);

        // 1. Single pass n-gram collection with optimized validation
        int historySize = simulationMoveHistory.size();
        for (int endIdx = 0; endIdx < historySize; endIdx++) {
            PNSMCTS_MAST.MoveWithPlayer record = simulationMoveHistory.get(endIdx);
            int currentPlayer = record.player;

            // Get or create player's n-gram map
            Map<PNSMCTS_MAST.NGramKey, Integer> playerNGramCounts = aggregatedPathNGramCounts
                    .computeIfAbsent(currentPlayer, k -> new HashMap<>());

            // Process 1-grams (faster path for most common case)
            if (nGramSize >= 1) {
                PNSMCTS_MAST.NGramKey singleMoveKey = new PNSMCTS_MAST.NGramKey(Collections.singletonList(record.move));
                playerNGramCounts.merge(singleMoveKey, 1, Integer::sum);
            }

            // Process n-grams of length 2 to nGramSize
            if (nGramSize >= 2) {
                int maxN = Math.min(nGramSize, endIdx + 1);
                sequenceBuilder.clear();

                // Build sequences in reverse order (from current move backwards)
                for (int n = 2; n <= maxN; n++) {
                    int startIdx = endIdx - n + 1;
                    if (startIdx < 0) continue; // Not enough history for this n-gram size

                    // Check if the sequence ends with the current player's move
                    if (simulationMoveHistory.get(endIdx).player != currentPlayer) {
                        continue; // Skip if the last move is not from the current player
                    }

                    // Build the sequence and validate player alternation
                    sequenceBuilder.clear();
                    boolean validSequence = true;

                    // Check player alternation pattern
                    for (int i = 0; i < n; i++) {
                        int currentIdx = startIdx + i;
                        PNSMCTS_MAST.MoveWithPlayer currentRecord = simulationMoveHistory.get(currentIdx);
                        sequenceBuilder.add(currentRecord.move);

                        // The last move must be from the current player
                        // The player should alternate with each move in the sequence
                        boolean shouldBeCurrentPlayer = (i % 2 == (n - 1) % 2);
                        if ((currentRecord.player == currentPlayer) != shouldBeCurrentPlayer) {
                            validSequence = false;
                            break;
                        }
                    }

                    if (validSequence) {
                        PNSMCTS_MAST.NGramKey ngramKey = new PNSMCTS_MAST.NGramKey(new ArrayList<>(sequenceBuilder));
                        playerNGramCounts.merge(ngramKey, 1, Integer::sum);
                    }
                }
            }
        }

        // 2. Update statistics with minimal lock time
//        synchronized (statsLock) {


        // Process each player's n-grams
        for (Map.Entry<Integer, Map<PNSMCTS_MAST.NGramKey, Integer>> playerEntry : aggregatedPathNGramCounts.entrySet()) {
            int playerRole = playerEntry.getKey();
            boolean isAI = (playerRole == this.player);
            double utility = utilities[playerRole];

//                System.out.printf("\nProcessing player %d (AI: %b, utility: %.2f)%n", playerRole, isAI, utility);
//
//                // Print all n-grams found for this player
//                System.out.println("N-grams found for player " + playerRole + ":");
//                for (Map.Entry<NGramKey, Integer> ngramEntry : playerEntry.getValue().entrySet()) {
//                    NGramKey ngram = ngramEntry.getKey();
//                    System.out.printf("  %d-gram [%d occurrences]: ",
//                            ngram.sequence.size(), ngramEntry.getValue());
//                    for (Move m : ngram.sequence) {
//                        System.out.print(m.getActionsWithConsequences(contextEnd) + " -> ");
//                    }
//                    System.out.println();
//                }

            // Select appropriate maps
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Double>> scoresMap = isAI ? nGramScores : opponentNGramScores;
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Integer>> visitsMap = isAI ? nGramVisits : opponentNGramVisits;
            Map<PNSMCTS_MAST.NGramKey, Integer> lastSeenMap = isAI ? nGramLastSeen : opponentNGramLastSeen;
            Map<Move, Integer> moveVisitsMap = isAI ? aiMoveVisits : opponentMoveVisits;
            Map<Move, Double> moveScoresMap = isAI ? aiMoveScores : opponentMoveScores;

            // Process each n-gram for this player
            for (Map.Entry<PNSMCTS_MAST.NGramKey, Integer> ngramEntry : playerEntry.getValue().entrySet()) {
                PNSMCTS_MAST.NGramKey ngram = ngramEntry.getKey();
                int occurrences = ngramEntry.getValue();
                Move move = ngram.sequence.get(ngram.sequence.size() - 1);

                // Get or create score and visit maps for this move
                Map<PNSMCTS_MAST.NGramKey, Double> moveScores = scoresMap.computeIfAbsent(move, k -> new HashMap<>());
                Map<PNSMCTS_MAST.NGramKey, Integer> moveVisits = visitsMap.computeIfAbsent(move, k -> new HashMap<>());

                // Update last seen and track additions
                lastSeenMap.put(ngram, currentTotalSimulations);
                boolean isNewNGram = !moveScores.containsKey(ngram);
                if (isNewNGram) nGramAdditions++;

                // Get current statistics
                double currentScore = moveScores.getOrDefault(ngram, 0.0);
                int currentVisits = moveVisits.getOrDefault(ngram, 0);
                int newVisits = currentVisits + occurrences;

                // Weighted average update: (currentScore * currentVisits + utility * occurrences) / (currentVisits + occurrences)
                double newScore = (currentScore * currentVisits + utility * occurrences) / newVisits;

                // Update maps
                moveScores.put(ngram, newScore);
                moveVisits.put(ngram, newVisits);

                // Update move-level statistics
                updateMoveStats(move, moveScores, moveVisits, moveScoresMap, moveVisitsMap);
                // Apply decay if needed
                if (doDecay && (currentTotalSimulations % DECAY_INTERVAL == 0) && isAI) {
                    applySimulationDecay(nGramScores, nGramVisits, ngram, occurrences);
                }else if(doDecay && (currentTotalSimulations % DECAY_INTERVAL == 0) && !isAI){
                    applySimulationDecay(opponentNGramScores, opponentNGramVisits, ngram, occurrences);
                }
            }
        }
//        }
    }

    // Helper method to apply decay to score and visit maps for a specific NGramKey
    // The decay is applied based on the number of occurrences of the NGramKey
    private void applySimulationDecay(
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Double>> scoresMap,
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Integer>> visitsMap,
            PNSMCTS_MAST.NGramKey nGramKey,
            int occurrences
    ) {
//        System.out.println("simulationdecay in progress");
        // Calculate the decay factor raised to the power of occurrences
        double decayFactor = Math.pow(simulationDecayFactor, occurrences);

        // Apply decay to scores and visits for the specific NGramKey
        for (Map<PNSMCTS_MAST.NGramKey, Double> scores : scoresMap.values()) {
            scores.computeIfPresent(nGramKey, (k, v) -> v * decayFactor);
        }
        for (Map<PNSMCTS_MAST.NGramKey, Integer> visits : visitsMap.values()) {
            visits.computeIfPresent(nGramKey, (k, v) -> (int)Math.ceil(v * decayFactor));
        }
    }

    // Helper method to apply decay to score and visit maps
    private void applyMoveDecay(
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Double>> scoresMap,
            Map<Move, Map<PNSMCTS_MAST.NGramKey, Integer>> visitsMap
    ) {
//        System.out.println("movedecay in progress");
        for (Map<PNSMCTS_MAST.NGramKey, Double> scores : scoresMap.values()) {
            scores.replaceAll((k, v) -> v * moveDecayFactor);
        }
        for (Map<PNSMCTS_MAST.NGramKey, Integer> visits : visitsMap.values()) {
            visits.replaceAll((k, v) -> (int)Math.ceil(v * moveDecayFactor));
        }
    }

    // Helper method to update move-level statistics
    private void updateMoveStats(
            Move move,
            Map<PNSMCTS_MAST.NGramKey, Double> scoresMap,
            Map<PNSMCTS_MAST.NGramKey, Integer> visitsMap,
            Map<Move, Double> moveScoresMap,
            Map<Move, Integer> moveVisitsMap
    ) {
        double totalScore = 0.0;
        int validNGrams = 0;

        // Single pass through n-grams for this move
        for (Map.Entry<PNSMCTS_MAST.NGramKey, Double> entry : scoresMap.entrySet()) {
            PNSMCTS_MAST.NGramKey ngram = entry.getKey();
            int visits = visitsMap.getOrDefault(ngram, 0);

            // Only consider n-grams with sufficient visits (for n > 1)
            if (visits > 0 && (ngram.sequence.size() == 1 || visits >= MIN_VISITS)) {
                totalScore += entry.getValue();
                validNGrams++;
            }
        }

        // Update move statistics if we have valid n-grams
        if (validNGrams > 0) {
            moveScoresMap.put(move, totalScore);
            moveVisitsMap.put(move, moveVisitsMap.getOrDefault(move, 0) + 1);
        }
    }




    /**
     * Resets this instance's N-gram statistics and move history.
     * Call this when starting a new match or when you want to clear this instance's statistics.
     */
    public void resetNGramStats() {
//        synchronized (statsLock) {
//            // Clear all n-gram statistics
//            nGramScores.clear();
//            nGramVisits.clear();
//            nGramLastSeen.clear();
//            opponentNGramScores.clear();
//            opponentNGramLastSeen.clear();
//            opponentNGramVisits.clear();
//            aiMoveScores.clear();
//            aiMoveVisits.clear();
//            opponentMoveScores.clear();
//            opponentMoveVisits.clear();
//
//            // Reset instance counters
//            totalSimulations = 0;
//            nGramAdditions = 0;
//        }
//
//        // Clear the simulation move history
//        simulationMoveHistory.clear();

        // Suggest garbage collection to free up memory
        //System.gc();
    }

    /**
     * Cleans up all instance-specific data and resets N-gram statistics.
     * Call this between games or when reusing the AI instance.
     *
     * Note: This only affects the current instance's data.
     */
    /**
     * Cleans up all instance-specific data and resets N-gram statistics.
     * Call this between games or when reusing the AI instance.
     * <p>
     * Note: This only affects the current instance's data.
     */
    public void cleanup() {
        if (printDebug) {
            System.out.println("Cleaning up AI state... nothing is done becouase its commanded out becaouse of the static field");
        }

        // Reset N-gram statistics (which also clears the move history)
        //resetNGramStats();

        // Reset any other instance-specific state
//        sims = 0;
//        simsThisTurn = 0;
//        turns = 0;
//        player = -1;

        if (printDebug) {
            System.out.println("AI state cleanup complete. nothing is done becouase its commanded out becaouse of the static field");
        }
    }

}
