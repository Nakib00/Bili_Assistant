// Initialize the map
var map = L.map('map', {
    minZoom: -5,
    maxZoom: 5,
    center: [0, 0],
    zoom: 50
});

// Load the house floor plan as an image layer 
var bounds = [
    [0, 0],
    [500, 500]
]; // Adjust size based on your image
var imageUrl = 'static/images/IUBMap.png'; 
var image = L.imageOverlay(imageUrl, bounds).addTo(map);

// Set view to the bounds of the image
map.fitBounds(bounds);

// Coordinates for points
const points = {
    yourPosition: [40, 320],
    auditorium: [63, 408],
    multipurposeHall: [74, 410],
    informationDesk: [70, 385],
    lobby: [78, 395],
    admissionOffice: [41, 377],
    helloCanteen: [82, 350],
    dosaOffice: [79, 260],
    souvenirShop: [79, 245],
    jolilShop: [77, 231],
    proctorOffice: [74, 231],
    washroom: [82, 226],
    foodCourt: [81, 209],
    swimmingPool: [73, 155],
    dmkBuilding: [55, 130],
    jubileeBuilding: [68, 80],
    securityBox: [42, 298],
    HealthCenter: [82, 277]
};

// Add marker for "Your Position"
L.marker(points.yourPosition, {
        title: 'Your Position'
    })
    .addTo(map)
    .bindPopup('Your Position')
    .openPopup(); // Automatically open the popup when the map loads

// Add markers for the other points
Object.keys(points).forEach(function (key) {
    if (key !== 'yourPosition') {
        L.marker(points[key]).addTo(map).bindPopup(key.replace(/([A-Z])/g, ' $1').trim());
    }
});

// Custom route example with waypoints
const customRouteauditorium = [
    points.yourPosition,
    [45, 320],
    [45, 388],
    [50, 388],
    [70, 388],
    [70, 408],
    points.auditorium // Final destination
];
const customRouteMultipurposeHall = [
    points.yourPosition,
    [45, 320],
    [45, 388],
    [50, 388],
    [70, 388],
    [70, 408],
    points.multipurposeHall // Final destination
];
const customRouteinformationDesk = [
    points.yourPosition,
    [45, 320],
    [45, 388],
    [50, 388],
    [70, 388],
    points.informationDesk // Final destination
];
const customRoutelobby = [
    points.yourPosition,
    [45, 320],
    [45, 388],
    [50, 388],
    points.lobby // Final destination
];
const customRouteadmissionOffice = [
    points.yourPosition,
    [45, 320],
    [45, 377],
    points.admissionOffice // Final destination
];
const customRoutehelloCanteen = [
    points.yourPosition,
    [47, 320],
    [47, 280],
    [81, 280],
    points.helloCanteen // Final destination
];
const customRouteHealthCenter = [
    points.yourPosition,
    [47, 320],
    [47, 280],
    points.HealthCenter // Final destination
];
const customRoutedosaOffice = [
    points.yourPosition,
    [47, 320],
    [47, 280],
    [80, 280],
    points.dosaOffice // Final destination
];
const customRoutesouvenirShop = [
    points.yourPosition,
    [47, 320],
    [47, 280],
    [81, 280],
    points.souvenirShop // Final destination
];
const customRoutejolilShop = [
    points.yourPosition,
    [47, 320],
    [47, 235],
    [75, 225],
    points.jolilShop // Final destination
];
const customRouteproctorOffice = [
    points.yourPosition,
    [47, 320],
    [47, 235],
    points.proctorOffice // Final destination
];
const customRoutewashroom = [
    points.yourPosition,
    [47, 320],
    [47, 230],
    points.washroom // Final destination
];
const customRoutefoodCourt = [
    points.yourPosition,
    [47, 320],
    [47, 230],
    [81, 225],
    points.foodCourt // Final destination
];
const customRouteswimmingPool = [
    points.yourPosition,
    [47, 320],
    [47, 250],
    [65, 225],
    [65, 155],
    points.swimmingPool // Final destination
];
const customRoutedmkBuilding = [
    points.yourPosition,
    [47, 320],
    [47, 250],
    [65, 225],
    [65, 155],
    points.dmkBuilding // Final destination
];
const customRoutejubileeBuilding = [
    points.yourPosition,
    [47, 320],
    [47, 250],
    [65, 225],
    [65, 155],
    [65, 85],
    points.jubileeBuilding // Final destination
];
const customRoutesecurityBox = [
    points.yourPosition,
    [47, 320],
    points.securityBox // Final destination
];
// Routes between points
const routes = {
    yourPositionToAuditorium: customRouteauditorium,
    yourPositionToMultipurposeHall: customRouteMultipurposeHall,
    yourPositionToInformationDesk: customRouteinformationDesk,
    yourPositionToLobby: customRoutelobby,
    yourPositionToAdmissionOffice: customRouteadmissionOffice,
    yourPositionToHelloCenter: customRoutehelloCanteen,
    yourPositionToDosaOffice: customRoutedosaOffice,
    yourPositionToSouvenirShop: customRoutesouvenirShop,
    yourPositionToJolilShop: customRoutejolilShop,
    yourPositionToProctorOffice: customRouteproctorOffice,
    yourPositionToWashroom: customRoutewashroom,
    yourPositionToFoodCourt: customRoutefoodCourt,
    yourPositionToSwimmingPool: customRouteswimmingPool,
    yourPositionToDmkBuilding: customRoutedmkBuilding,
    yourPositionToJubileeBuilding: customRoutejubileeBuilding,
    yourPositionToSecurityBox: customRoutesecurityBox,
    yourPositionToHealthCenter: customRouteHealthCenter
};

// Event listener for the button click to show the route
document.getElementById('show-route').addEventListener('click', function () {
    var endPoint = document.getElementById('end-point').value;

    // Log the selected endpoint
    console.log("Selected endpoint:", endPoint);

    // Remove any existing route from the map
    if (window.routeLine) {
        map.removeLayer(window.routeLine);
    }
    if (window.routeDecorator) {
        map.removeLayer(window.routeDecorator);
    }

    // Check if the end point is selected
    if (endPoint) {
        var customRouteKey = 'yourPositionTo' + endPoint.charAt(0).toUpperCase() + endPoint.slice(1);
        var routeCoordinates = routes[customRouteKey];

        // Log the route coordinates
        console.log("Route coordinates:", routeCoordinates);

        // Draw the polyline (route)
        window.routeLine = L.polyline(routeCoordinates, {
            color: 'blue',
            weight: 5
        }).addTo(map);

        // Create an animated route with arrows
        window.routeDecorator = L.polylineDecorator(window.routeLine, {
            patterns: [{
                offset: '100%',
                repeat: 0,
                symbol: L.Symbol.arrowHead({
                    pixelSize: 15,
                    polygon: false,
                    pathOptions: {
                        stroke: true,
                        color: 'red',
                        weight: 2
                    }
                })
            }]
        }).addTo(map);

        // Animate the arrow moving along the path
        let offset = 0;
        setInterval(() => {
            offset = (offset + 1) % 100;
            window.routeDecorator.setPatterns([{
                offset: `${offset}%`,
                repeat: 0,
                symbol: L.Symbol.arrowHead({
                    pixelSize: 15,
                    polygon: false,
                    pathOptions: {
                        stroke: true,
                        color: 'red',
                        weight: 2
                    }
                })
            }]);
        }, 100); // Speed of animation

        // Fit the map bounds to show the entire route
        map.fitBounds(window.routeLine.getBounds());
    } else {
        alert('Please select a destination point.');
    }
});

// Function to get query parameters
function getQueryParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// On page load, check for destination parameter
document.addEventListener('DOMContentLoaded', function() {
    const destination = getQueryParameter('destination');
    if (destination) {
        // Set the dropdown to the specified destination
        const selectElement = document.getElementById('end-point');
        selectElement.value = destination;

        // Optionally, trigger the route display automatically
        document.getElementById('show-route').click();
    }
});
