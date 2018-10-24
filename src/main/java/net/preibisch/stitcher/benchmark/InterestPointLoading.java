package net.preibisch.stitcher.benchmark;


import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import mpicbg.models.AffineModel3D;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.NoninvertibleModelException;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.RigidModel3D;
import mpicbg.models.TranslationModel3D;
import mpicbg.spim.data.SpimData;
import mpicbg.spim.data.SpimDataException;
import mpicbg.spim.data.sequence.Channel;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.mpicbg.PointMatchGeneric;
import net.imglib2.util.Util;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoint;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPointList;
import net.preibisch.mvrecon.process.interestpointregistration.TransformationTools;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.PairwiseResult;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.fastrgldm.FRGLDMMatcher;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.fastrgldm.FRGLDMPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.fastrgldm.FRGLDMParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.icp.IterativeClosestPointPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.icp.IterativeClosestPointParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.ransac.RANSACParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.rgldm.RGLDMPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.rgldm.RGLDMParameters;

public class InterestPointLoading
{

	private static class FieldParameters
	{
		public List<List<Integer>> points;
		//public List<Integer> off;
	}
	
	private static class BiobeamSimulationParameters
	{
		// NB: we don't need to specify all elements from JSON
		/*
		public String save_fstring;
		public Integer seed;
		public String raw_data_path;
		public List<Integer> raw_data_dims;
		public List<Integer> phys_dims;
		public Double na_illum;
		public Double na_detect;
		public Double ri_medium;
		public List<Double> ri_delta_range;
		public Boolean two_sided_illum;
		public List<Integer> lambdas;
		public List<Integer> fov_size;
		public List<Integer> point_fov_size;
		public Integer n_points_per_fov;
		public Integer points_min_distance;
		public List<Integer> min_off;
		public List<Integer> max_off;
		public List<Integer> x_locs;
		public List<Integer> y_locs;
		public List<Integer> z_locs;
		public Integer padding;
		public List<Integer> conv_subblocks;
		public Integer bprop_nvolumes;
		*/
		public Integer downsampling;
		public Map<String, FieldParameters> fields;
	}

	public static Map<String, List<List<Integer>>> getGTCoordinatesFromJSON(String simulationParameterFile)
	{
		Map<String, List<List<Integer>>> res = new HashMap<>();

		// load JSON simulation parameters
		Gson gson = new GsonBuilder().setPrettyPrinting().create();
		BiobeamSimulationParameters params = null;
		try (FileReader fr = new FileReader( simulationParameterFile ))
		{
			params = gson.fromJson( fr, BiobeamSimulationParameters.class);
		}
		catch ( IOException e) { 
			e.printStackTrace();
			return null;
		}

		// extract just coordinates, correct for downsampling (coordinates are saved in full res)
		final int ds = params.downsampling;
		params.fields.forEach( (k, v) -> res.put( k,
				v.points.stream().map(
						c -> c.stream().map( ci -> ci / ds ).collect( Collectors.toList() )
						).collect( Collectors.toList() ) ));
		return res;
	}
	
	public static List<InterestPoint> getInterestPointsFromSpimData(SpimData2 data, int channelId, String label)
	{
		// get all vIDs for channel
		final ArrayList< ViewId > allVids = new ArrayList<>(data.getSequenceDescription().getViewDescriptions().keySet());
		final ArrayList< ? extends ViewId > allViewIdsForChannelSorted = SpimData2.getAllViewIdsForChannelSorted( data, allVids, new Channel( channelId ) );

		// get all IPs of given label for vIDs
		final Map< ViewId, List< InterestPoint > > allInterestPoints = TransformationTools.getAllInterestPoints( allViewIdsForChannelSorted,
					data.getViewRegistrations().getViewRegistrations(),
					data.getViewInterestPoints().getViewInterestPoints(),
					allViewIdsForChannelSorted.stream().collect( Collectors.toMap( v -> v, v -> label ) ), true );

		// un-group, we don't need this information a.t.m.
		final List< InterestPoint > ungroupedIPs = allInterestPoints.values().stream().reduce(new ArrayList<InterestPoint>(), (a,b) -> {
			ArrayList<InterestPoint> res = new ArrayList<InterestPoint>();
			res.addAll( a );
			res.addAll( b );
			return res;
		});

		return ungroupedIPs;
	}

	public static ArrayList< InterestPoint > gtCoordinatesToIPs(List< List< Integer > > locsFirst, boolean invertCoordinates)
	{
		// Integer-list lists to InterestPoint lists
		ArrayList<InterestPoint> ipsFirst = new ArrayList<>();
		int id = 0;
		for (List<Integer> loc : locsFirst)
		{
			double[] x = new double[loc.size()];
			for (int i=0; i<loc.size(); i++)
			{
				// gt coords might be zyx
				x[i] = loc.get(invertCoordinates ? loc.size() - i - 1 : i );
			}
			ipsFirst.add( new InterestPoint( id++, x ) );
		}
		return ipsFirst;
	}

	public static void main(String[] args)
	{

		SpimData2 data = null;
		try
		{
			data = new XmlIoSpimData2( "" ).load( "/Volumes/davidh-ssd/mv-sim/sim3/intensity_adjusted/dataset.xml" );
		}
		catch ( SpimDataException e ) { e.printStackTrace(); }
		final List< InterestPoint > ipsAll = getInterestPointsFromSpimData( data, 2, "beads" );

		// coordinates from simulation parameters
		final Map< String, List< List< Integer > > > gtCoords = getGTCoordinatesFromJSON( "/Volumes/davidh-ssd/mv-sim/sim3/sim3.json" );
		List<List<Integer>> locsFirst = new ArrayList<>();
		locsFirst.addAll( gtCoords.get( "0,3,0" ) );
		ArrayList< InterestPoint > ipsFirst = gtCoordinatesToIPs( locsFirst, true );
		
		// match with ransac
		final RANSACParameters ransacParameters = new RANSACParameters();
		// TODO: RANSAC should have 10/10 inliers, but does not
		// probably a numerical issue, since there is NO error in the GT
		ransacParameters.setMinInlierFactor( 2f );
		final RGLDMParameters dp = new RGLDMParameters( new AffineModel3D(), Float.MAX_VALUE, 2.0f, 3, 2 );
		RGLDMPairwise< InterestPoint > matcherPairwise = new RGLDMPairwise<>( ransacParameters, dp );
		//final FRGLDMParameters matcherParameters = new FRGLDMParameters( new AffineModel3D(), 2f, 2 );
		//FRGLDMPairwise< InterestPoint > matcherPairwise = new FRGLDMPairwise<>( ransacParameters, matcherParameters );
		PairwiseResult< InterestPoint > match = matcherPairwise.match( ipsFirst, ipsAll );
		System.out.println( match.getFullDesc() );
		for (PointMatchGeneric< InterestPoint > pm : match.getInliers())
			System.out.println( Util.printCoordinates( pm.getPoint1() ) + " " + Util.printCoordinates( pm.getPoint2() ));

		AffineModel3D model = new AffineModel3D();
		try
		{
			model.fit( match.getInliers() );
			System.out.println( model );
		}
		catch ( NotEnoughDataPointsException | IllDefinedDataPointsException e )
		{
			e.printStackTrace();
		}

		// match without ransac, just FRGLD
		/*
		FRGLDMMatcher< InterestPoint > matcher2 = new FRGLDMMatcher<>();
		ArrayList< PointMatchGeneric< InterestPoint > > match2 = matcher2.extractCorrespondenceCandidates( ipsFirst, ipsAll, 0, 5 );
		for (PointMatchGeneric< InterestPoint > pm : match2)
			System.out.println( Util.printCoordinates( pm.getPoint1() ) + " " + Util.printCoordinates( pm.getPoint2() ));
		*/
		
		
		// refine the alignment via ICP
		IterativeClosestPointParameters icpParams = new IterativeClosestPointParameters( new AffineModel3D() );
		IterativeClosestPointPairwise< InterestPoint > icp = new IterativeClosestPointPairwise< InterestPoint >( icpParams );
		ArrayList< InterestPoint > ipAllTransformed = new ArrayList<>();

		// transform one set by inverse of first round
		for (final InterestPoint ip : ipsAll)
		{
			InterestPoint ipTrans = null;
			try
			{
				ipTrans = new InterestPoint( ip.getId(), model.applyInverse( ip.getW() ) );
			}
			catch ( NoninvertibleModelException e )
			{
				e.printStackTrace();
			}
			//System.out.println( Util.printCoordinates( ipTrans ) + ", " + Util.printCoordinates( ip ) );
			ipAllTransformed.add( ipTrans );
		}

		PairwiseResult< InterestPoint > match3 = icp.match( ipsFirst, ipAllTransformed );
		for (PointMatchGeneric< InterestPoint > pm : match3.getInliers())
			System.out.println( Util.printCoordinates( pm.getPoint1() ) + " " + Util.printCoordinates( pm.getPoint2() ));
		
		
		
		
		
		//System.out.println( data.getViewInterestPoints().getViewInterestPoints());
	}


	
	
}
