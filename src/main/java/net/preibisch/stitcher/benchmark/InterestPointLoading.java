package net.preibisch.stitcher.benchmark;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import mpicbg.models.AffineModel3D;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.InverseCoordinateTransform;
import mpicbg.models.Model;
import mpicbg.models.NoninvertibleModelException;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.spim.data.SpimDataException;
import mpicbg.spim.data.sequence.Channel;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.mpicbg.PointMatchGeneric;
import net.imglib2.util.Pair;
import net.imglib2.util.Util;
import net.imglib2.util.ValuePair;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoint;
import net.preibisch.mvrecon.process.interestpointregistration.TransformationTools;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.PairwiseResult;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.fastrgldm.FRGLDMPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.fastrgldm.FRGLDMParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.icp.IterativeClosestPointPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.icp.IterativeClosestPointParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.ransac.RANSACParameters;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.rgldm.RGLDMPairwise;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.rgldm.RGLDMParameters;

public class InterestPointLoading
{

	private static class OutPointMatch
	{
		public double[] gt;
		public double[] data;
	}

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
	
	public static <I extends InterestPoint> List< Pair< I, I > > matchGTtoDataRGLDM(
			final List<I> gtPoints,
			final List<I> dataPoints,
			final RANSACParameters rp,
			final RGLDMParameters mp,
			final IterativeClosestPointParameters ip
			)
	{
		RGLDMPairwise< I > matcherPairwise = new RGLDMPairwise<>( rp, mp );
		PairwiseResult< I > match = matcherPairwise.match( gtPoints, dataPoints );
		
		Model<?> model =  mp.getModel().copy();
		try
		{
			model.fit( match.getInliers() );
		}
		catch ( NotEnoughDataPointsException | IllDefinedDataPointsException e )
		{
			return null;
		}
		
		ArrayList< InterestPoint > ipDataTransformed = new ArrayList<>();

		// transform one set by inverse of first round
		for (final I ipnt : dataPoints)
		{
			InterestPoint ipntTrans = null;
			try
			{
				ipntTrans = new InterestPoint( ipnt.getId(), ( (InverseCoordinateTransform) model ).applyInverse( ipnt.getW() ) );
			}
			catch ( NoninvertibleModelException e )
			{
				return null;
			}
			ipDataTransformed.add( ipntTrans );
		}
		
		IterativeClosestPointPairwise< InterestPoint > icp = new IterativeClosestPointPairwise< InterestPoint >( ip );

		PairwiseResult< InterestPoint > match2 = icp.match( (List< InterestPoint >) gtPoints, ipDataTransformed );
		List< Pair< I, I > > res = new ArrayList<>();
		for (PointMatchGeneric< InterestPoint > pm : match2.getInliers())
		{
			InterestPoint ipntTrans = null;
			try
			{
				ipntTrans = new InterestPoint( pm.getPoint2().getId(), ( (InverseCoordinateTransform) model ).applyInverse( pm.getPoint2().getW() ) );
			}
			catch ( NoninvertibleModelException e )
			{
				return null;
			}
			res.add( new ValuePair< I, I >( (I) pm.getPoint1(), (I)ipntTrans ) );
		}
		return res;
	}
	
	
	public static <I extends InterestPoint> List< Pair< I, I > > matchGTtoDataFRGLDM(
			final List<I> gtPoints,
			final List<I> dataPoints,
			final RANSACParameters rp,
			final FRGLDMParameters mp,
			final IterativeClosestPointParameters ip
			)
	{
		FRGLDMPairwise< I > matcherPairwise = new FRGLDMPairwise<>( rp, mp );
		PairwiseResult< I > match = matcherPairwise.match( gtPoints, dataPoints );
		
		Model<?> model =  mp.getModel().copy();
		try
		{
			model.fit( match.getInliers() );
		}
		catch ( NotEnoughDataPointsException | IllDefinedDataPointsException e )
		{
			return null;
		}
		
		ArrayList< InterestPoint > ipDataTransformed = new ArrayList<>();

		// transform one set by inverse of first round
		for (final I ipnt : dataPoints)
		{
			InterestPoint ipntTrans = null;
			try
			{
				ipntTrans = new InterestPoint( ipnt.getId(), ( (InverseCoordinateTransform) model ).applyInverse( ipnt.getW() ) );
			}
			catch ( NoninvertibleModelException e )
			{
				return null;
			}
			ipDataTransformed.add( ipntTrans );
		}
		
		IterativeClosestPointPairwise< InterestPoint > icp = new IterativeClosestPointPairwise< InterestPoint >( ip );

		PairwiseResult< InterestPoint > match2 = icp.match( (List< InterestPoint >) gtPoints, ipDataTransformed );
		List< Pair< I, I > > res = new ArrayList<>();
		for (PointMatchGeneric< InterestPoint > pm : match2.getInliers())
		{
			InterestPoint ipntTrans = null;
			try
			{
				ipntTrans = new InterestPoint( pm.getPoint2().getId(), ( (InverseCoordinateTransform) model ).applyInverse( pm.getPoint2().getW() ) );
			}
			catch ( NoninvertibleModelException e )
			{
				return null;
			}
			res.add( new ValuePair< I, I >( (I) pm.getPoint1(), (I)ipntTrans ) );
		}
		return res;
	}

	public static void main(String[] args)
	{

		String datasetFilePath =  "/Volumes/davidh-ssd/mv-sim/sim5/intensity_adjusted/dataset.xml";
		String gtFilePath =  "/Volumes/davidh-ssd/mv-sim/sim5/sim5.json";
		String outFilePath =  "/Volumes/davidh-ssd/mv-sim/sim5/match_results.json";
		
		// coordinates detected in SpimData
		SpimData2 data = null;
		try
		{
			data = new XmlIoSpimData2( "" ).load( datasetFilePath );
		}
		catch ( SpimDataException e ) { e.printStackTrace(); }
		final List< InterestPoint > ipsAll = getInterestPointsFromSpimData( data, 2, "beads" );

		// coordinates from simulation parameters
		final Map< String, List< List< Integer > > > gtCoords = getGTCoordinatesFromJSON( gtFilePath );

		// parameters for matchers
		final RANSACParameters rp = new RANSACParameters();
		rp.setMinInlierFactor( 1.5f );
		final RGLDMParameters mp = new RGLDMParameters( new AffineModel3D(), Float.MAX_VALUE, 1.5f, 3, 2 );
		final IterativeClosestPointParameters ip = new IterativeClosestPointParameters( new AffineModel3D() );
		
		Map<String, List<OutPointMatch>> out = new HashMap<>();
		
		for (String field : gtCoords.keySet())
		{
			System.out.println( " === " + field + " === " );
			final List<List<Integer>> locsGTfield = new ArrayList<>();
			locsGTfield.addAll( gtCoords.get( field ) );
			final ArrayList< InterestPoint > ipsFirst = gtCoordinatesToIPs( locsGTfield, true );
			List< Pair< InterestPoint, InterestPoint > > res = matchGTtoDataRGLDM( ipsFirst, ipsAll, rp, mp, ip );
			if (res == null)
				System.out.println( "NO match found." );
			else
			{
				System.out.println( "Found " + res.size() + " matching points.");
				List< OutPointMatch > resOut = res.stream().map( pm -> {
					OutPointMatch opm = new OutPointMatch();
					opm.gt = pm.getA().getW();
					opm.data = pm.getB().getW();
					return opm;
				} ).collect( Collectors.toList() );
				out.put( field, resOut );
			}	
			
		}
		
		final Gson gson = new GsonBuilder().setPrettyPrinting().create();
		
		try ( PrintWriter pr = new PrintWriter( new File( outFilePath ) ))
		{
			pr.print( gson.toJson( out ) );
		}
		catch ( FileNotFoundException e )
		{
			e.printStackTrace();
		}
		System.out.println( gson.toJson( out ) );
	}


	
	
}
