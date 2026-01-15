package main;

import app.StartDesktopApp;
import manager.ai.AIRegistry;
import mcts.*;
import search.mcts.MCTS;

/**
 * The main method of this launches the Ludii application with its GUI, and registers
 * the example AIs from this project such that they are available inside the GUI.
 *
 * @author Dennis Soemers
 */
public class LaunchLudii
{
	/**
	 * The main method
	 * @param args
	 */
	public static void main(final String[] args)
	{
		// params to test:
		boolean finMove = true;
		int minVisits = 5;
		double pnCons = 1.0;
		double cFactor = 0.2;

		// Register our example AIs
		if (!AIRegistry.registerAI("Example Standard UTC AI", () -> MCTS.createUCT(), (game) -> true))
			System.err.println("WARNING! Failed to register AI because one with that name already existed!");

		double[] setting = {1, Math.sqrt(2), 1};
		if (!AIRegistry.registerAI("PN UCT", () -> new PNSMCTS_L2(setting), (game) -> new PNSMCTS_L2(setting).supportsGame(game)))
			System.err.println("WARNING! Failed to register AI because one with that name already existed!");

		//PNSMCTS_L2_MAST david celik
		// Create an instance of your AI with 2-gram NST
		PNSMCTS_L2_MAST mastAI = new PNSMCTS_L2_MAST(finMove, minVisits, pnCons, cFactor, 3);  // Using 2-gram NST
		if (!AIRegistry.registerAI("PNS_L2_MAST UCT", () -> mastAI,
				(game) -> mastAI.supportsGame(game))) {
			System.err.println("WARNING! Failed to register AI because one with that name already existed!");
		}

		// PNSMCTS_L2_RAVE with RAVE
		double[] raveSettings = {1, Math.sqrt(2), 1}; // [PN-constant, UCT-constant, time-per-turn, RAVE_K]
		PNSMCTS_L2_RAVE raveAI = new PNSMCTS_L2_RAVE(finMove, minVisits, pnCons, cFactor);
		if (!AIRegistry.registerAI("PNS_L2_RAVE UCT", () -> raveAI,
				(game) -> raveAI.supportsGame(game))) {
			System.err.println("WARNING! Failed to register RAVE AI because one with that name already existed!");
		}

		// Add a shutdown hook to print final statistics
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			System.out.println("\n=== Final N-gram Statistics ===");
			mastAI.printNGramStats(10, null);  // Print top 10 N-grams of each size (null context as this is a shutdown hook)
		}));
		
		// Run Ludii
		StartDesktopApp.main(new String[0]);
	}
}
